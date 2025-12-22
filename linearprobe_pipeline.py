from encoders import get_encoder, get_features
from datasets import get_dataset
from online_variance import WelfordOnlineVariance, Normalization
from torch.utils.data.dataloader import DataLoader
import torch , os
from torch.functional import F
from tqdm.notebook import tqdm
import numpy as np
import json, re
from enum import Enum

class BoostingMethod(Enum):
    D_GRADIENTS = "dimming_gradients"
    B_GRADIENTS = "boosting_gradients"
    DROP_OUT = "drop_out"
    WEIGHTS = "weights"
    WEIGHTS_PENALTY = "weights_penalty"

def save_checkpoint(path, classifier, optimizer, epoch, history, hyperparams, variance_tracker= None, weights_only=False):
    checkpoint = {
        'classifier_state': classifier.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch,
        'history': history,
        'hyperparams': hyperparams,
    }
    if variance_tracker:
        checkpoint['variance_tracker'] = {'n': variance_tracker.n, 'mean': variance_tracker.mean, 'M2': variance_tracker.M2}
    torch.save(checkpoint, path)

def load_checkpoint(path, classifier, optimizer, variance_tracker= None):
    checkpoint = torch.load(path, weights_only=False)
    classifier.load_state_dict(checkpoint['classifier_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    if variance_tracker is not None and 'variance_tracker' in checkpoint:
        vt_state = checkpoint['variance_tracker']
        variance_tracker.n = vt_state['n']
        variance_tracker.mean = vt_state['mean']
        variance_tracker.M2 = vt_state['M2']
    epoch = checkpoint['epoch']
    history = checkpoint['history']
    return classifier, optimizer, epoch, history, variance_tracker

class GradHooker:
    def __init__(self):
        self.weights = None
        self.hooking_method = None

    def set_dimming(self, weights):
        self.weights = weights
        self.hooking_method = "dimming"

    def set_boosting(self, weights, percentile_threshold, scale):
        self.weights = weights
        self.percentile_threshold = percentile_threshold
        self.scale = scale
        self.hooking_method = "boosting"

    def hook(self, grad):
        w = self.weights.unsqueeze(0)
        if self.hooking_method == "dimming":
            pass
        elif self.hooking_method == "boosting":
            threshold = torch.quantile(self.weights, self.percentile_threshold)
            boost_mask = self.weights >= threshold
            w[:, boost_mask] = w[:, boost_mask] * self.scale
        return grad * w

def parse_exp_filename(filename):

    parts = filename.split("_")

    escaped_encoder_name = parts[0]
    dataset_name = parts[1]
    mode = parts[2]
    if mode == "V":
        return {
            "encoder_name": escaped_encoder_name,
            "dataset_name": dataset_name,
            "boost_with_variance": False,
            "variance_tracker_window": None,
            "boosting_active_threshold": None,
            "variance_normalization": None,
            "boosting_method": None,
            "boosting_percentile_threshold": None,
            "boosting_scale": None,
        }

    def extract(token):
        return token[token.index("(") + 1 : token.index(")")]

    variance_tracker_window = int(extract(parts[3]))
    boosting_active_threshold = float(extract(parts[4]))

    variance_normalization = Normalization[parts[5].replace("-", "_")]

    boosting_method = BoostingMethod[parts[6].replace("-", "_")]

    boosting_percentile_threshold = float(extract(parts[7])) if extract(parts[7]) else None
    boosting_scale = float(extract(parts[8])) if extract(parts[8]) else None

    return {
        "encoder_name": escaped_encoder_name,
        "dataset_name": dataset_name,
        "boost_with_variance": True,
        "variance_tracker_window": variance_tracker_window,
        "boosting_active_threshold": boosting_active_threshold,
        "variance_normalization": variance_normalization,
        "boosting_method": boosting_method,
        "boosting_percentile_threshold": boosting_percentile_threshold,
        "boosting_scale": boosting_scale,
    }


def get_exp_filename(encoder_name, dataset_name, boost_with_variance, variance_tracker_window,
                     boosting_active_threshold, variance_normalization, boosting_method,
                     boosting_percentile_threshold, boosting_scale):
    escaped_encoder_name = encoder_name.replace("/", "-")
    if boost_with_variance:
        variance_tracker_window_name = f"vtw({variance_tracker_window})"
        boosting_active_threshold_name = f"bathre({boosting_active_threshold})"
        normalization_method_name = variance_normalization.name.replace("_", "-")
        boosting_method_name = boosting_method.name.replace("_", "-")
        boosting_percentile_threshold_name = f"bpt({boosting_percentile_threshold})"
        boosting_scale_name = f"bs({boosting_scale})"
        chkpt_filename = f"{escaped_encoder_name}_{dataset_name}_B_{variance_tracker_window_name}_{boosting_active_threshold_name}_{normalization_method_name}_{boosting_method_name}_{boosting_percentile_threshold_name}_{boosting_scale_name}"
    else:
        chkpt_filename = f"{escaped_encoder_name}_{dataset_name}_V"
    return chkpt_filename

def probe(encoder_name, dataset_name, boost_with_variance= False, batch_size= 64, n_epochs= 20,
          encoder_target_dim=768, num_workers=4, learning_rate=1e-3, variance_tracker_window=10,
          boosting_active_threshold=100, variance_normalization=Normalization.MIN_MAX, boosting_method = BoostingMethod.D_GRADIENTS,
          boosting_percentile_threshold=85, boosting_scale=1.5, optimizer_type= "adam",
          random_state=42, chkpt_path="./chkpt", test_every_x_steps=1, validate= False,
          verbose=True):
    
    # Set random seed for reproducibility
    torch.manual_seed(random_state)
    
    # Save hyperparameters
    hyperparams = {
        "batch_size": batch_size,
        "encoder_target_dim": encoder_target_dim,
        "learning_rate": learning_rate,
    }
    
    # Create checkpoint directory
    if not os.path.exists(chkpt_path):
        os.mkdir(chkpt_path)

    # Get encoder
    if verbose: print("Loading model ...")
    encoder, processor = get_encoder(encoder_name)

    # Get device
    device = next(encoder.parameters()).device
    
    # Get datasets
    if verbose: print("Loading dataset ...")
    train_dataset = get_dataset(dataset_name, "train", processor)
    test_dataset = get_dataset(dataset_name, "test", processor)
    val_dataset = get_dataset(dataset_name, "val", processor)

    # Get dataloaders
    if verbose: print("Loading dataloaders ...")
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle= True, num_workers= num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle= False, num_workers= num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle= False, num_workers= num_workers)

    if verbose: print("Setting up online varience weighting ...")
    if boost_with_variance:
        variance_tracker = WelfordOnlineVariance(encoder_target_dim, active_threshold=boosting_active_threshold, moving_average_window=variance_tracker_window, normalization=variance_normalization, device= device)
        if "gradients" in boosting_method.value.lower():
            grad_booster = GradHooker()
        else:
            None
    else:
        variance_tracker = None
        grad_booster = None

    # Define classifier
    if verbose: print("Defining classifier ...")
    classifier = torch.nn.Linear(encoder_target_dim, train_dataset.num_labels())
    if boost_with_variance and "gradients" in boosting_method.value.lower():
        classifier.weight.register_hook(grad_booster.hook)
    classifier.to(device)

    # Define optimizer
    if verbose: print("Defining optimizer ...")
    if optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(classifier.parameters(), lr=learning_rate)
    elif optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    # Load checkpoint
    if verbose: print("Loading checkpoint ...")
    chkpt_filename = get_exp_filename(encoder_name, dataset_name, boost_with_variance,
                                      variance_tracker_window, boosting_active_threshold,
                                      variance_normalization, boosting_method,
                                      boosting_percentile_threshold, boosting_scale)
    chkpt_filepath = os.path.join(chkpt_path, f"{chkpt_filename}.pt")
    if os.path.exists(chkpt_filepath):
        classifier, optimizer, start_epoch, history, variance_tracker = load_checkpoint(chkpt_filepath, classifier, optimizer, variance_tracker) 
    else:
        start_epoch = 0
        history = []

    # Define criterion
    if verbose: print("Defining criterion ...")
    if train_dataset.is_multilabel():
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if verbose: print("Starting training ...")
    for epoch in range(start_epoch, n_epochs):
        train_losses = []

        classifier.train()
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{n_epochs}')
        for batch in pbar:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                features = get_features(encoder, inputs, encoder_target_dim, device="cuda")
            if boost_with_variance:
                variance_tracker.update(features)
                var_weights = variance_tracker.variance_weights()
                _log_vars(variance_tracker.variance(), chkpt_path, f"{chkpt_filename}_var_logs")
                if boosting_method == BoostingMethod.D_GRADIENTS:
                    grad_booster.set_dimming(var_weights)
                    outputs = classifier(features)
                elif boosting_method == BoostingMethod.B_GRADIENTS:
                    grad_booster.set_boosting(var_weights, boosting_percentile_threshold, boosting_scale)
                    outputs = classifier(features)
                elif boosting_method == BoostingMethod.WEIGHTS:
                    weights = var_weights.view(1, -1)
                    weighted_weights = classifier.weight * weights * boosting_scale
                    outputs = F.linear(features, weighted_weights, classifier.bias)
                    outputs = classifier(features)
                elif boosting_method == BoostingMethod.DROP_OUT:
                    threshold = torch.quantile(var_weights, boosting_percentile_threshold)
                    drop_mask = var_weights < threshold
                    features[:, drop_mask] = 0
                    outputs = classifier(features)
                elif boosting_method == BoostingMethod.WEIGHTS_PENALTY:
                    low_threshold = torch.quantile(var_weights, boosting_percentile_threshold[0])
                    high_threshold = torch.quantile(var_weights, boosting_percentile_threshold[1])
                    low_var_weights = classifier.weight[:, var_weights < low_threshold]
                    high_var_weights = classifier.weight[:, var_weights > high_threshold]
                    penalty = boosting_scale * (low_var_weights.pow(2).sum() / (high_var_weights.pow(2).sum() + 1e-8))
                    outputs = classifier(features)
                else:
                    raise Exception("Not supported boosting method.")
                _log_vars(var_weights, chkpt_path, f"{chkpt_filename}_var_logs_weights")
            else:
                outputs = classifier(features)
            if boost_with_variance and boosting_method == BoostingMethod.WEIGHTS_PENALTY:
                loss = criterion(outputs, labels) + penalty
            else:
                loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            pbar.set_postfix({"Train Loss": loss.item()})

        train_loss = sum(train_losses) / len(train_losses)
        tqdm.write(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}")

        # Validation loop
        if validate:
            classifier.eval()
            val_losses = []
            val_preds = []
            val_labels = []
            pbar = tqdm(val_dataloader, desc=f'Validation Epoch {epoch+1}/{n_epochs}')
            for batch in pbar:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)  
                
                with torch.no_grad():
                    features = get_features(encoder, inputs, encoder_target_dim, device="cuda")
                
                if boost_with_variance and boosting_method == BoostingMethod.WEIGHTS:
                    var_weights = variance_tracker.variance_weights().view(1, -1)
                    weights = var_weights.view(1, -1)
                    weighted_weights = classifier.weight * weights * boosting_scale
                    outputs = F.linear(features, weighted_weights, classifier.bias)
                    outputs = classifier(features)
                else:
                    outputs = classifier(features)

                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                pbar.set_postfix({"Val Loss": loss.item()})

                if train_dataset.is_multilabel():
                    predicted = (torch.sigmoid(outputs) > 0.5).int()
                    val_preds.extend(predicted.flatten().cpu().numpy())
                    val_labels.extend(labels.flatten().cpu().numpy())
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            val_loss = sum(val_losses) / len(val_losses)
            val_acc = 100.0 * (np.array(val_preds) == np.array(val_labels)).sum() / len(val_labels)
            tqdm.write(f"Epoch {epoch+1}/{n_epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
        else:
            val_loss = None
            val_acc = None
            
        # Testing loop
        if (epoch+1) % test_every_x_steps == 0:
            classifier.eval()
            test_losses = []
            test_preds = []
            test_labels = []
            pbar = tqdm(test_dataloader, desc=f'Testing Epoch {epoch+1}/{n_epochs}')
            for batch in pbar:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                with torch.no_grad():
                    features = get_features(encoder, inputs, encoder_target_dim, device="cuda")
                
                if boost_with_variance and boosting_method == BoostingMethod.WEIGHTS:
                    var_weights = variance_tracker.variance_weights().view(1, -1)
                    weights = var_weights.view(1, -1)
                    weighted_weights = classifier.weight * weights * boosting_scale
                    outputs = F.linear(features, weighted_weights, classifier.bias)
                    outputs = classifier(features)
                else:
                    outputs = classifier(features)

                loss = criterion(outputs, labels)
                test_losses.append(loss.item())
                pbar.set_postfix({"Test Loss": loss.item()})

                if train_dataset.is_multilabel():
                    predicted = (torch.sigmoid(outputs) > 0.5).int()
                    test_preds.extend(predicted.flatten().cpu().numpy())
                    test_labels.extend(labels.flatten().cpu().numpy())
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    test_preds.extend(predicted.cpu().numpy())
                    test_labels.extend(labels.cpu().numpy())
            
            test_loss = sum(test_losses) / len(test_losses)
            test_acc = 100.0 * (np.array(test_preds) == np.array(test_labels)).sum() / len(test_labels)
            tqdm.write(f"Epoch {epoch+1}/{n_epochs}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
        else:
            test_loss = None
            test_acc = None

        # Save checkpoint
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc
        })

        save_checkpoint(chkpt_filepath, classifier, optimizer, epoch + 1, history, hyperparams, variance_tracker)

def _log_vars(var, path="./", filename="val_logs"):
    var = var.tolist()
    log_vars = bool(os.environ.get("LOG_VARIANCE", "False"))
    if log_vars:
        path = os.path.join(path, f"{filename}.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                prev_vars = json.load(f)
        else:
            prev_vars = []
        prev_vars.append(var)
        with open(path, "w") as f:
            json.dump(prev_vars, f)
    