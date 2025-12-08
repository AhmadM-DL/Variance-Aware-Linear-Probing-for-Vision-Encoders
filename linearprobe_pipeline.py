from encoders import get_encoder, get_features
from datasets import get_dataset
from online_variance import WelfordOnlineVariance
from torch.utils.data.dataloader import DataLoader
import torch , os, json
from time import time
from tqdm.notebook import tqdm
import numpy as np

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

def probe(encoder_name, dataset_name, boost_gradients_with_variance= False, batch_size= 64, n_epochs= 20,
          encoder_target_dim=768, num_workers=4, learning_rate=1e-3,
          random_state=42, chkpt_path="./chkpt",
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
    if boost_gradients_with_variance:
        variance_tracker = WelfordOnlineVariance(encoder_target_dim, device= encoder.device)
    else:
        variance_tracker = None

    # Define classifier
    if verbose: print("Defining classifier ...")
    classifier = torch.nn.Linear(encoder_target_dim, train_dataset.num_labels())
    classifier.to(encoder.device)

    # Define optimizer
    if verbose: print("Defining optimizer ...")
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)

    # Load checkpoint
    if verbose: print("Loading checkpoint ...")
    escaped_encoder_name = encoder_name.replace("/", "_")
    escaped_dataset_name = dataset_name.replace("/", "_")
    boosted = "boosted" if boost_gradients_with_variance else "vanilla"
    chkpt_filename = f"{escaped_encoder_name}_{escaped_dataset_name}_{boosted}.pt"
    chkpt_filepath = os.path.join(chkpt_path, chkpt_filename)
    if os.path.exists(chkpt_filepath):
        classifier, optimizer, start_epoch, history, variance_tracker = load_checkpoint(chkpt_filepath, classifier, optimizer, variance_tracker) 
    else:
        start_epoch = 0
        history = []

    # Define criterion
    if verbose: print("Defining criterion ...")
    reduction = "none" if boost_gradients_with_variance else "mean"
    if train_dataset.is_multilabel():
        criterion = torch.nn.BCEWithLogitsLoss(reduction = reduction)
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction = reduction)

    if verbose: print("Starting training ...")
    for epoch in range(start_epoch, n_epochs):
        train_losses = []

        classifier.train()
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{n_epochs}')
        for batch in pbar:
            inputs, labels = batch
            inputs = inputs.to(encoder.device)
            labels = labels.to(encoder.device)
            with torch.no_grad():
                features = get_features(encoder, inputs, encoder_target_dim, device="cuda")
            if boost_gradients_with_variance:
                variance_tracker.update(features)
            outputs = classifier(features)
            loss_vector = criterion(outputs, labels)
            if boost_gradients_with_variance:
                var_weights = variance_tracker.variance_weights().view(1, -1)
                weighted_loss =  loss_vector*var_weights
                loss = weighted_loss.mean()
            else:
                loss = loss_vector
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            pbar.set_postfix({"Train Loss": loss.item()})

        train_loss = sum(train_losses) / len(train_losses)
        tqdm.write(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}")

        # Validation loop
        classifier.eval()
        val_losses = []
        val_preds = []
        val_labels = []
        pbar = tqdm(val_dataloader, desc=f'Validation Epoch {epoch+1}/{n_epochs}')
        for batch in pbar:
            inputs, labels = batch
            inputs = inputs.to(encoder.device)
            labels = labels.to(encoder.device)  
            
            with torch.no_grad():
                features = get_features(encoder, inputs, encoder_target_dim, device="cuda")
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

        # Testing loop
        if (epoch+1) % 5 == 0:
            classifier.eval()
            test_losses = []
            test_preds = []
            test_labels = []
            pbar = tqdm(test_dataloader, desc=f'Testing Epoch {epoch+1}/{n_epochs}')
            for batch in pbar:
                inputs, labels = batch
                inputs = inputs.to(encoder.device)
                labels = labels.to(encoder.device)
                
                with torch.no_grad():
                    features = get_features(encoder, inputs, encoder_target_dim, device="cuda")
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
