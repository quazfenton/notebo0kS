"""
Sliced Wasserstein Distance & Gradient Flows
Efficient optimal transport for high-dimensional distributions
Applications: generative models, domain adaptation, distribution matching
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class SlicedWassersteinDistance(nn.Module):
    """
    Compute Sliced Wasserstein distance between distributions
    
    SW(μ, ν) = ∫_{S^(d-1)} W₁(P_θμ, P_θν) dθ
    
    where P_θ is projection onto direction θ, W₁ is 1D Wasserstein distance
    """
    
    def __init__(self, num_projections: int = 100, p: int = 2):
        super().__init__()
        self.num_projections = num_projections
        self.p = p  # Order of Wasserstein distance
        
    def forward(
        self,
        x: torch.Tensor,  # Samples from distribution μ: [batch_size, dim]
        y: torch.Tensor   # Samples from distribution ν: [batch_size, dim]
    ) -> torch.Tensor:
        """
        Compute sliced Wasserstein distance between x and y
        
        Args:
            x: Samples from source distribution [n_samples, dim]
            y: Samples from target distribution [m_samples, dim]
            
        Returns:
            Sliced Wasserstein distance (scalar)
        """
        batch_size, dim = x.shape
        device = x.device
        
        # Sample random projections from unit sphere
        projections = self._random_projections(dim, self.num_projections, device)
        
        # Project distributions onto random directions
        x_proj = x @ projections  # [n_samples, num_projections]
        y_proj = y @ projections  # [m_samples, num_projections]
        
        # Compute 1D Wasserstein distance for each projection
        sw_distance = 0.0
        for i in range(self.num_projections):
            # Sort projected samples
            x_sorted, _ = torch.sort(x_proj[:, i])
            y_sorted, _ = torch.sort(y_proj[:, i])
            
            # 1D Wasserstein distance (closed-form for sorted samples)
            w1_distance = torch.mean(torch.abs(x_sorted - y_sorted) ** self.p)
            sw_distance += w1_distance
        
        sw_distance = (sw_distance / self.num_projections) ** (1.0 / self.p)
        
        return sw_distance
    
    def _random_projections(
        self,
        dim: int,
        num_projections: int,
        device: torch.device
    ) -> torch.Tensor:
        """Sample uniform random directions on unit sphere"""
        # Gaussian samples normalized to unit sphere
        projections = torch.randn(dim, num_projections, device=device)
        projections = projections / torch.norm(projections, dim=0, keepdim=True)
        return projections


class GeneralizedSlicedWasserstein(nn.Module):
    """
    Generalized Sliced Wasserstein with learned projections
    Can learn task-specific slicing directions
    """
    
    def __init__(
        self,
        dim: int,
        num_projections: int = 50,
        learnable_projections: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.num_projections = num_projections
        
        if learnable_projections:
            # Initialize with random projections, make learnable
            init_projections = torch.randn(dim, num_projections)
            init_projections = init_projections / torch.norm(init_projections, dim=0, keepdim=True)
            self.projections = nn.Parameter(init_projections)
        else:
            self.register_buffer(
                'projections',
                self._random_projections(dim, num_projections)
            )
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Normalize projection vectors to unit sphere
        normalized_proj = self.projections / torch.norm(
            self.projections, dim=0, keepdim=True
        )
        
        # Project
        x_proj = x @ normalized_proj
        y_proj = y @ normalized_proj
        
        # Compute SW distance
        sw_distance = 0.0
        for i in range(self.num_projections):
            x_sorted, _ = torch.sort(x_proj[:, i])
            y_sorted, _ = torch.sort(y_proj[:, i])
            
            w1_distance = torch.mean(torch.abs(x_sorted - y_sorted))
            sw_distance += w1_distance
        
        return sw_distance / self.num_projections
    
    def _random_projections(self, dim: int, num_proj: int) -> torch.Tensor:
        proj = torch.randn(dim, num_proj)
        return proj / torch.norm(proj, dim=0, keepdim=True)


class SlicedWassersteinFlow(nn.Module):
    """
    Neural network that learns transport map via sliced Wasserstein gradient flow
    
    Minimize: SW(G_θ(μ), ν)
    where G_θ pushes source distribution μ toward target ν
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dims: list = [128, 128],
        num_projections: int = 100
    ):
        super().__init__()
        
        self.dim = dim
        
        # Neural transport map: x → G(x)
        layers = []
        prev_dim = dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, dim))
        
        self.transport_map = nn.Sequential(*layers)
        
        # Sliced Wasserstein loss
        self.sw_distance = SlicedWassersteinDistance(num_projections)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transport map to samples"""
        return self.transport_map(x)
    
    def transport_loss(
        self,
        source_samples: torch.Tensor,
        target_samples: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss for learning transport map
        
        Args:
            source_samples: Samples from source distribution μ
            target_samples: Samples from target distribution ν
            
        Returns:
            Sliced Wasserstein distance between G(μ) and ν
        """
        transported = self.transport_map(source_samples)
        loss = self.sw_distance(transported, target_samples)
        return loss


class MaxSlicedWasserstein(nn.Module):
    """
    Max-Sliced Wasserstein: maximize over projections for tighter bound
    
    MSW(μ, ν) = max_{θ ∈ S^(d-1)} W₁(P_θμ, P_θν)
    """
    
    def __init__(self, dim: int, num_iterations: int = 10, lr: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_iterations = num_iterations
        self.lr = lr
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute max sliced Wasserstein via gradient ascent on projection
        
        Returns:
            max_distance: Maximum Wasserstein distance
            max_projection: Direction achieving maximum
        """
        device = x.device
        
        # Initialize random projection
        theta = torch.randn(self.dim, 1, device=device, requires_grad=True)
        theta = theta / torch.norm(theta)
        
        optimizer = torch.optim.SGD([theta], lr=self.lr)
        
        for _ in range(self.num_iterations):
            optimizer.zero_grad()
            
            # Normalize to unit sphere
            theta_normalized = theta / torch.norm(theta)
            
            # Project and compute 1D Wasserstein
            x_proj = (x @ theta_normalized).squeeze()
            y_proj = (y @ theta_normalized).squeeze()
            
            x_sorted, _ = torch.sort(x_proj)
            y_sorted, _ = torch.sort(y_proj)
            
            # Maximize this distance
            distance = torch.mean(torch.abs(x_sorted - y_sorted))
            
            # Gradient ascent
            (-distance).backward()
            optimizer.step()
        
        with torch.no_grad():
            theta_final = theta / torch.norm(theta)
            x_proj = (x @ theta_final).squeeze()
            y_proj = (y @ theta_final).squeeze()
            x_sorted, _ = torch.sort(x_proj)
            y_sorted, _ = torch.sort(y_proj)
            max_distance = torch.mean(torch.abs(x_sorted - y_sorted))
        
        return max_distance, theta_final


def example_distribution_matching():
    """
    Example: learn transport map between two Gaussian distributions
    """
    torch.manual_seed(42)
    
    dim = 10
    num_samples = 1000
    
    # Source: standard Gaussian
    source_mean = torch.zeros(dim)
    source_samples = torch.randn(num_samples, dim)
    
    # Target: shifted and scaled Gaussian
    target_mean = torch.ones(dim) * 2.0
    target_cov_factor = torch.randn(dim, dim) * 0.5
    target_samples = (
        torch.randn(num_samples, dim) @ target_cov_factor.T + target_mean
    )
    
    # Create transport model
    model = SlicedWassersteinFlow(dim, hidden_dims=[64, 64], num_projections=50)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        loss = model.transport_loss(source_samples, target_samples)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, SW Distance: {loss.item():.4f}")
    
    # Evaluate learned transport
    with torch.no_grad():
        transported_samples = model(source_samples)
        
        print(f"\nSource mean: {source_samples.mean(dim=0)[:3].numpy()}")
        print(f"Target mean: {target_samples.mean(dim=0)[:3].numpy()}")
        print(f"Transported mean: {transported_samples.mean(dim=0)[:3].numpy()}")
        
        # Final distance
        sw = SlicedWassersteinDistance(num_projections=100)
        final_distance = sw(transported_samples, target_samples)
        print(f"\nFinal SW distance: {final_distance.item():.6f}")


def example_max_sliced_wasserstein():
    """
    Demonstrate max-sliced Wasserstein computation
    """
    torch.manual_seed(42)
    
    dim = 5
    x = torch.randn(200, dim)
    y = torch.randn(200, dim) + 1.0  # Shifted distribution
    
    # Standard sliced Wasserstein
    sw = SlicedWassersteinDistance(num_projections=100)
    sw_dist = sw(x, y)
    print(f"Sliced Wasserstein (100 projections): {sw_dist.item():.4f}")
    
    # Max sliced Wasserstein
    msw = MaxSlicedWasserstein(dim, num_iterations=20, lr=0.1)
    msw_dist, max_proj = msw(x, y)
    print(f"Max Sliced Wasserstein: {msw_dist.item():.4f}")
    print(f"Worst-case projection direction: {max_proj.squeeze().numpy()}")


if __name__ == "__main__":
    print("=== Distribution Matching Example ===")
    example_distribution_matching()
    
    print("\n=== Max Sliced Wasserstein Example ===")
    example_max_sliced_wasserstein()
