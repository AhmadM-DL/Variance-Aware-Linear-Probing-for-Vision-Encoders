import torch
from torch.functional import F
from enum import Enum, auto

class VarianceWeightingStrategy(Enum):
    FEATURE_MULTIPLY = auto()
    LOSS_MULTIPLY = auto()

class WelfordOnlineVariance:
    def __init__(self, num_features, strategy, active_threshold=200, device='cuda'):
        self.n = 0
        self.mean = torch.zeros(num_features, device=device)
        self.M2 = torch.zeros(num_features, device=device)
        self.strategy = strategy
        self.active_threshold = active_threshold

    @torch.no_grad()
    def update(self, x):
        """
        x: tensor of shape (batch_size, num_features)
        Updates running mean and variance.
        """
        batch_size = x.shape[0]
        self.n += batch_size

        # batch mean
        batch_mean = x.mean(dim=0)
        # difference between batch mean and running mean
        delta = batch_mean - self.mean

        # update running mean
        self.mean += delta * batch_size / self.n

        # update M2 (sum of squared deviations)
        batch_var = x.var(dim=0, unbiased=False)
        self.M2 += batch_var * batch_size + delta**2 * (batch_size * (self.n - batch_size) / self.n)

    def variance(self):
        if self.n < 2:
            return torch.zeros_like(self.mean)
        return self.M2 / (self.n - 1)
    
    def variance_weights(self):
        if self.n < self.active_threshold:
            return torch.ones_like(self.mean)
        var = self.variance() 
        weights = F.softmax(var, dim=0)
        return weights
    
    def apply_weights(self, vector):
        weights = self.variance_weights()
        return vector * weights

    def std(self):
        return torch.sqrt(self.variance())    

def _test_apply_weights():
    torch.manual_seed(123)
    num_samples = 500
    num_features = 10
    batch_size = 50
    data = torch.randn(num_samples, num_features) * 2 + 3
    data = data.to("cuda")
    tracker = WelfordOnlineVariance(num_features=num_features)
    idx = 0
    while idx < num_samples:
        batch = data[idx:idx+batch_size]
        tracker.update(batch)
        weighted_batch = tracker.apply_weights(batch)
        assert weighted_batch.shape == batch.shape, "Weighted batch has wrong shape"
        assert not torch.isnan(weighted_batch).any(), "NaNs in weighted batch"
        if tracker.n < 20:
            weights = tracker.variance_weights()
            max_diff = (weights - 1/num_features).abs().max().item()
            assert max_diff < 0.2, f"Weights too peaky early: max diff {max_diff}"
        idx += batch_size
    
def _test_welford_online_variance():
    torch.manual_seed(42)
    num_samples = 1_000
    num_features = 768
    batch_size = 137 
    data = torch.randn(num_samples, num_features) * 2 + 3
    data = data.to("cuda")
    tracker = WelfordOnlineVariance(num_features=num_features)
    idx = 0
    while idx < num_samples:
        batch = data[idx:idx+batch_size]
        tracker.update(batch)
        idx += batch_size
    true_var = data.var(dim=0, unbiased=True)
    online_var = tracker.variance()
    max_abs_diff = (true_var - online_var).abs().max().item()
    mean_abs_diff = (true_var - online_var).abs().mean().item()
    print("=== Welford Online Variance Test ===")
    print(f"Max abs diff   : {max_abs_diff:.8f}")
    print(f"Mean abs diff  : {mean_abs_diff:.8f}")
    assert max_abs_diff < 1e-4