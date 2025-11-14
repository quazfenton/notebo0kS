"""
Equivariant Message Passing with Attention (EMPA)
Graph Neural Network with guaranteed geometric equivariance
Applicable to molecules, proteins, point clouds, physics simulations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class EquivariantMessagePassing(nn.Module):
    """
    EMPA Layer: Message passing with equivariance to rotation and translation
    
    For group G acting on features, ensures: f(g·x) = g·f(x) for all g ∈ G
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int = 0,
        num_heads: int = 4,
        dropout: float = 0.0,
        equivariance_type: str = 'rotation'  # 'rotation', 'translation', 'permutation'
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        self.dropout = dropout
        self.equivariance_type = equivariance_type
        
        assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"
        
        # Attention mechanism
        self.q_proj = nn.Linear(in_channels, out_channels)
        self.k_proj = nn.Linear(in_channels, out_channels)
        self.v_proj = nn.Linear(in_channels, out_channels)
        
        # Edge features integration
        if edge_dim > 0:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_dim, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels)
            )
        else:
            self.edge_encoder = None
        
        # Equivariant transformation (group representation)
        if equivariance_type == 'rotation':
            # SO(3) equivariance via vector features
            self.invariant_proj = nn.Linear(in_channels, out_channels)
            self.equivariant_proj = nn.Linear(in_channels, out_channels)
        
        # Message aggregation
        self.message_mlp = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels)
        )
        
        # Output projection
        self.output_mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels)
        )
        
        self.layer_norm = nn.LayerNorm(out_channels)
        
    def forward(
        self,
        x: torch.Tensor,  # Node features [num_nodes, in_channels]
        edge_index: torch.Tensor,  # Edges [2, num_edges]
        edge_attr: Optional[torch.Tensor] = None,  # Edge features [num_edges, edge_dim]
        pos: Optional[torch.Tensor] = None  # 3D positions [num_nodes, 3] for geometric equivariance
    ) -> torch.Tensor:
        """
        Forward pass with equivariant message passing
        
        Args:
            x: Node feature matrix
            edge_index: Graph connectivity [source_nodes, target_nodes]
            edge_attr: Optional edge features
            pos: Optional 3D positions for geometric equivariance
            
        Returns:
            Updated node features
        """
        num_nodes = x.size(0)
        
        # Multi-head attention queries, keys, values
        q = self.q_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        
        # Message passing
        messages = []
        
        source_nodes, target_nodes = edge_index[0], edge_index[1]
        
        # Compute attention scores
        q_i = q[target_nodes]  # [num_edges, num_heads, head_dim]
        k_j = k[source_nodes]  # [num_edges, num_heads, head_dim]
        v_j = v[source_nodes]  # [num_edges, num_heads, head_dim]
        
        # Scaled dot-product attention
        attn_scores = (q_i * k_j).sum(dim=-1) / math.sqrt(self.head_dim)  # [num_edges, num_heads]
        
        # Incorporate edge features into attention
        if edge_attr is not None and self.edge_encoder is not None:
            edge_embed = self.edge_encoder(edge_attr)  # [num_edges, out_channels]
            edge_embed = edge_embed.view(-1, self.num_heads, self.head_dim)
            attn_scores = attn_scores + (q_i * edge_embed).sum(dim=-1)
        
        # Geometric equivariance: distance-based attention modulation
        if pos is not None:
            rel_pos = pos[target_nodes] - pos[source_nodes]  # [num_edges, 3]
            dist = torch.norm(rel_pos, dim=-1, keepdim=True)  # [num_edges, 1]
            
            # Distance-aware attention (maintains translation equivariance)
            dist_weight = torch.exp(-dist / 5.0)  # Learnable scale in practice
            attn_scores = attn_scores * dist_weight
        
        # Softmax over incoming edges per node
        attn_weights = self._scatter_softmax(attn_scores, target_nodes, num_nodes)  # [num_edges, num_heads]
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Aggregate messages with attention
        weighted_v = v_j * attn_weights.unsqueeze(-1)  # [num_edges, num_heads, head_dim]
        
        # Rotation equivariance: separate scalar and vector features
        if self.equivariance_type == 'rotation' and pos is not None:
            # Scalar features (invariant)
            scalar_msg = weighted_v.view(-1, self.out_channels)
            aggregated_scalar = self._scatter_sum(scalar_msg, target_nodes, num_nodes)
            
            # Vector features (equivariant)
            rel_pos_normalized = F.normalize(rel_pos, dim=-1)  # [num_edges, 3]
            vector_msg = self.equivariant_proj(x[source_nodes]) * attn_weights.mean(dim=1, keepdim=True)
            vector_msg = vector_msg.unsqueeze(-1) * rel_pos_normalized.unsqueeze(1)  # [num_edges, out_channels, 3]
            aggregated_vector = self._scatter_sum(vector_msg, target_nodes, num_nodes)
            
            # Combine invariant and equivariant parts
            aggregated_norm = torch.norm(aggregated_vector, dim=-1)  # [num_nodes, out_channels]
            aggregated = torch.cat([aggregated_scalar, aggregated_norm], dim=-1)
            aggregated = self.message_mlp(aggregated)
        else:
            # Standard aggregation
            weighted_v = weighted_v.view(-1, self.out_channels)
            aggregated = self._scatter_sum(weighted_v, target_nodes, num_nodes)
        
        # Update with residual connection
        out = torch.cat([x, aggregated], dim=-1)
        out = self.message_mlp(out)
        
        # Residual + LayerNorm
        if x.size(-1) == out.size(-1):
            out = self.layer_norm(x + out)
        else:
            out = self.layer_norm(out)
        
        return out
    
    def _scatter_softmax(
        self,
        src: torch.Tensor,  # [num_edges, num_heads]
        index: torch.Tensor,  # [num_edges]
        num_nodes: int
    ) -> torch.Tensor:
        """Compute softmax over edges grouped by target node"""
        # Numerically stable softmax
        src_max = self._scatter_max(src, index, num_nodes)
        src_max = src_max[index]
        
        exp = torch.exp(src - src_max)
        exp_sum = self._scatter_sum(exp, index, num_nodes)
        exp_sum = exp_sum[index]
        
        return exp / (exp_sum + 1e-16)
    
    def _scatter_sum(
        self,
        src: torch.Tensor,
        index: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """Sum values with the same index"""
        out_shape = (num_nodes,) + src.shape[1:]
        out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
        return out.scatter_add_(0, index.unsqueeze(-1).expand_as(src), src)
    
    def _scatter_max(
        self,
        src: torch.Tensor,
        index: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """Max values with the same index"""
        out_shape = (num_nodes,) + src.shape[1:]
        out = torch.full(out_shape, float('-inf'), dtype=src.dtype, device=src.device)
        return out.scatter_reduce_(0, index.unsqueeze(-1).expand_as(src), src, reduce='amax')


class EquivariantGraphNetwork(nn.Module):
    """
    Full equivariant graph neural network with multiple EMPA layers
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        num_heads: int = 4,
        edge_dim: int = 0,
        dropout: float = 0.1,
        equivariance_type: str = 'rotation',
        pool: str = 'mean'  # 'mean', 'sum', 'max'
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.pool = pool
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # Stack of EMPA layers
        self.layers = nn.ModuleList([
            EquivariantMessagePassing(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                edge_dim=edge_dim,
                num_heads=num_heads,
                dropout=dropout,
                equivariance_type=equivariance_type
            ) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through equivariant GNN
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            pos: 3D positions [num_nodes, 3]
            batch: Batch assignment [num_nodes] for graph-level predictions
            
        Returns:
            Node-level or graph-level predictions
        """
        # Input projection
        x = self.input_proj(x)
        
        # Message passing layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, pos)
        
        # Graph-level pooling if batch is provided
        if batch is not None:
            if self.pool == 'mean':
                x = self._global_mean_pool(x, batch)
            elif self.pool == 'sum':
                x = self._global_sum_pool(x, batch)
            elif self.pool == 'max':
                x = self._global_max_pool(x, batch)
        
        # Output projection
        x = self.output_proj(x)
        
        return x
    
    def _global_mean_pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Average pooling over nodes in each graph"""
        num_graphs = batch.max().item() + 1
        out = torch.zeros(num_graphs, x.size(-1), dtype=x.dtype, device=x.device)
        
        for i in range(num_graphs):
            mask = batch == i
            out[i] = x[mask].mean(dim=0)
        
        return out
    
    def _global_sum_pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Sum pooling over nodes in each graph"""
        num_graphs = batch.max().item() + 1
        out = torch.zeros(num_graphs, x.size(-1), dtype=x.dtype, device=x.device)
        return out.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), x)
    
    def _global_max_pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Max pooling over nodes in each graph"""
        num_graphs = batch.max().item() + 1
        out = torch.full((num_graphs, x.size(-1)), float('-inf'), dtype=x.dtype, device=x.device)
        return out.scatter_reduce_(0, batch.unsqueeze(-1).expand_as(x), x, reduce='amax')


def example_molecule_classification():
    """Example: classify molecules using equivariant GNN"""
    
    # Simulate molecular graph data
    num_nodes = 20
    num_edges = 40
    in_channels = 16  # Atom features
    edge_dim = 4  # Bond features
    
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, edge_dim)
    pos = torch.randn(num_nodes, 3)  # 3D coordinates
    batch = torch.zeros(num_nodes, dtype=torch.long)  # Single graph
    
    # Create model
    model = EquivariantGraphNetwork(
        in_channels=in_channels,
        hidden_channels=64,
        out_channels=2,  # Binary classification
        num_layers=3,
        num_heads=4,
        edge_dim=edge_dim,
        equivariance_type='rotation'
    )
    
    # Forward pass
    output = model(x, edge_index, edge_attr, pos, batch)
    print(f"Output shape: {output.shape}")  # [1, 2]
    print(f"Predictions: {torch.softmax(output, dim=-1)}")
    
    # Test equivariance: rotation should not change predictions
    rotation_matrix = torch.tensor([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    pos_rotated = pos @ rotation_matrix.T
    output_rotated = model(x, edge_index, edge_attr, pos_rotated, batch)
    
    print(f"\nEquivariance test (should be small): {torch.norm(output - output_rotated).item():.6f}")


if __name__ == "__main__":
    example_molecule_classification()
