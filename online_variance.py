import torch
from collections import deque
from enum import Enum

class Normalization(Enum):
    NONE = "none"
    MIN_MAX = "min_max"
    SOFTMAX_T = "softmax_t"



class WelfordOnlineVariance:
    def __init__(self, num_features,
                active_threshold=200,
                moving_average_window= 10,
                normalization=Normalization.MIN_MAX,
                temperature=1.0,
                device='cuda'):
        
        self.n = 0
        self.mean = torch.zeros(num_features, device=device)
        self.M2 = torch.zeros(num_features, device=device)
        self.active_threshold = active_threshold
        self.moving_average_window = moving_average_window
        self.queue = deque(maxlen=moving_average_window)
        self.normalization = normalization
        self.temperute = temperature

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

        self.queue.append(self.variance())

    def variance(self):
        return self.M2 / (self.n - 1)  
    
    def moving_average_variance(self):
        return torch.stack(list(self.queue)).mean(dim=0)
    
    def variance_weights(self):
        if self.n < self.active_threshold:
            return torch.ones_like(self.mean)
        else:
            var = self.moving_average_variance()
            return _normalize(var, norm=self.normalization, temperature=self.temperature)

def _normalize(x, norm: Normalization, temperature: float = 1.0, eps: float = 1e-8):
    if norm == Normalization.NONE:
        return x

    if norm == Normalization.MIN_MAX:
        return (x - x.min()) / (x.max() - x.min() + eps)

    if norm == Normalization.SOFTMAX_T:
        return torch.softmax(x / temperature, dim=-1)

    raise ValueError(f"Unknown normalization: {norm}")

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