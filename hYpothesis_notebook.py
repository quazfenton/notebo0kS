#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scientific-Hypothesis Predictor Macro (SHPM)
=============================================
Advanced ML system for automated scientific hypothesis generation and experiment planning
with information-theoretic optimization and neurosymbolic architecture.
"""

# %% [markdown]
# # Scientific-Hypothesis Predictor Macro (SHPM)
# 
# ## Overview
# A neurosymbolic AI system that closes the scientific discovery loop by:
# - Generating hypotheses from prior experimental data
# - Recommending maximally informative experiments
# - Predicting outcomes under uncertainty
# - Learning causal patterns from scientific corpora

# %% Core Imports and Setup
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from torch_geometric.nn import GCNConv, global_mean_pool, GraphSAGE
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import networkx as nx
from collections import defaultdict, deque
import math
import json
from transformers import AutoModel, AutoTokenizer
from scipy.stats import entropy
from scipy.special import logsumexp
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# --- NBX pipeline (optional; no-op unless flags are provided) ---
if __name__ == "__main__":
    try:
        from nbx.pipeline import load_from_cli
        split, cfg = load_from_cli()
        if split is not None:
            logger.info(f"[NBX] Loaded dataset: {split.meta}")
    except Exception:
        pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %% [markdown]
# ## 1. Base Primitives and Knowledge Representation

# %% Hyperbolic Embeddings for Hierarchical Relations
class HyperbolicEmbedding(nn.Module):
    """
    Poincaré ball embeddings for hierarchical scientific taxonomies.
    Enables zero-shot extrapolation to new domains.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.c = 1.0  # Curvature
        
    def distance(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Hyperbolic distance in Poincaré ball"""
        norm_u = torch.clamp(torch.norm(u, dim=-1, keepdim=True), max=1-self.eps)
        norm_v = torch.clamp(torch.norm(v, dim=-1, keepdim=True), max=1-self.eps)
        
        dist = torch.norm(u - v, dim=-1)
        norm_prod = (1 - norm_u**2) * (1 - norm_v**2)
        
        return torch.acosh(1 + 2 * dist**2 / (norm_prod + self.eps))
    
    def exp_map(self, v: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Exponential map from tangent space to Poincaré ball"""
        norm_v = torch.norm(v, dim=-1, keepdim=True)
        norm_p = torch.norm(p, dim=-1, keepdim=True)
        
        lambda_p = 2 / (1 - norm_p**2 + self.eps)
        
        return torch.tanh(lambda_p * norm_v / 2) * v / (norm_v + self.eps)

# %% Scientific Knowledge Graph Module
@dataclass
class ScientificPrimitive:
    """Base primitive for scientific knowledge representation"""
    entity_type: str  # molecule, protein, parameter
    attributes: Dict[str, Any]
    embedding: Optional[torch.Tensor] = None
    
class KnowledgeGraphModule(nn.Module):
    """
    Knowledge graph for structured reasoning over scientific variables and relations.
    Integrates with transformer for hybrid neurosymbolic reasoning.
    """
    def __init__(self, entity_dim: int = 256, relation_dim: int = 128, 
                 num_layers: int = 3, use_hyperbolic: bool = True):
        super().__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.use_hyperbolic = use_hyperbolic
        
        # Entity and relation embeddings
        self.entity_embed = nn.Embedding(10000, entity_dim)  # Placeholder size
        self.relation_embed = nn.Embedding(100, relation_dim)
        
        # GraphSAGE for message passing
        self.conv_layers = nn.ModuleList([
            GraphSAGE(entity_dim if i == 0 else entity_dim * 2, 
                     entity_dim * 2, num_layers=1)
            for i in range(num_layers)
        ])
        
        # Hyperbolic embeddings for hierarchical relations
        if use_hyperbolic:
            self.hyperbolic = HyperbolicEmbedding(entity_dim)
        
        # Causal edge predictor with Bayesian weights
        self.edge_predictor = nn.Sequential(
            nn.Linear(entity_dim * 4 + relation_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # Mean and log-variance for Gaussian edge weights
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through knowledge graph"""
        
        # Message passing through graph
        for conv in self.conv_layers:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=0.2, training=self.training)
        
        # Global pooling if batch is provided
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x
    
    def predict_causal_edge(self, src: torch.Tensor, dst: torch.Tensor, 
                           relation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict probabilistic causal edge weights P(Y|X)"""
        combined = torch.cat([src, dst, src * dst, dst - src, relation], dim=-1)
        params = self.edge_predictor(combined)
        mean, log_var = params.chunk(2, dim=-1)
        return mean, torch.exp(log_var)

# %% [markdown]
# ## 2. Hybrid Transformer-KG Architecture

# %% Scientific Hypothesis Transformer
class ScientificTransformer(nn.Module):
    """
    Transformer for sequential hypothesis generation with KG cross-attention.
    Generates hypotheses, experimental designs, and outcome predictions.
    """
    def __init__(self, vocab_size: int = 50000, d_model: int = 768, 
                 n_heads: int = 12, n_layers: int = 6, kg_dim: int = 256):
        super().__init__()
        self.d_model = d_model
        
        # Token embeddings and positional encoding
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(512, d_model)
        
        # Transformer layers with KG cross-attention
        self.layers = nn.ModuleList([
            TransformerLayerWithKG(d_model, n_heads, kg_dim)
            for _ in range(n_layers)
        ])
        
        # Output heads
        self.hypothesis_head = nn.Linear(d_model, vocab_size)
        self.design_head = nn.Linear(d_model, 1024)  # Experimental design space
        self.outcome_head = VariationalHead(d_model, 256)  # VAE for outcome distribution
        
    def forward(self, input_ids: torch.Tensor, kg_features: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Generate hypothesis, design, and outcome predictions"""
        
        # Embed tokens
        seq_len = input_ids.size(1)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(pos_ids)
        
        # Pass through transformer layers with KG cross-attention
        for layer in self.layers:
            x = layer(x, kg_features, attention_mask)
        
        # Generate outputs
        outputs = {
            'hypothesis_logits': self.hypothesis_head(x),
            'design': torch.sigmoid(self.design_head(x[:, -1])),  # Last token for design
            'outcome_dist': self.outcome_head(x[:, -1])  # Outcome distribution params
        }
        
        return outputs

class TransformerLayerWithKG(nn.Module):
    """Transformer layer with knowledge graph cross-attention"""
    def __init__(self, d_model: int, n_heads: int, kg_dim: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.kg_proj = nn.Linear(kg_dim, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, x: torch.Tensor, kg_features: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        x = x + self.self_attn(x, x, x, attn_mask=mask)[0]
        x = self.norm1(x)
        
        # Cross-attention with KG
        kg_proj = self.kg_proj(kg_features)
        x = x + self.cross_attn(x, kg_proj, kg_proj)[0]
        x = self.norm2(x)
        
        # FFN
        x = x + self.ffn(x)
        x = self.norm3(x)
        
        return x

class VariationalHead(nn.Module):
    """VAE head for outcome distribution modeling"""
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim * 2)  # Mean and log-variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encode
        params = self.encoder(x)
        mean, log_var = params.chunk(2, dim=-1)
        
        # Reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + eps * std
        
        # Decode
        reconstruction = self.decoder(z)
        
        return {
            'mean': mean,
            'log_var': log_var,
            'z': z,
            'reconstruction': reconstruction
        }

# %% [markdown]
# ## 3. Information-Theoretic Reward System

# %% Reward and Loss Functions
class InformationGainReward:
    """
    Calculate information gain for experiment selection.
    Rewards experiments that maximally reduce posterior entropy.
    """
    def __init__(self, prior_entropy: float = 1.0):
        self.prior_entropy = prior_entropy
        
    def mutual_information(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Calculate mutual information I(X;Y) = H(Y) - H(Y|X)"""
        # Estimate entropies using kernel density estimation
        H_Y = self.estimate_entropy(Y)
        H_Y_given_X = self.conditional_entropy(Y, X)
        return H_Y - H_Y_given_X
    
    def estimate_entropy(self, X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Estimate entropy using histogram method"""
        X_flat = X.flatten()
        hist = torch.histc(X_flat, bins=50)
        hist = hist / hist.sum()
        hist = hist[hist > 0]  # Remove zeros
        return -torch.sum(hist * torch.log(hist + eps))
    
    def conditional_entropy(self, Y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Estimate H(Y|X) using conditional distributions"""
        # Simplified: use Gaussian assumption
        residuals = Y - X.mean(dim=-1, keepdim=True)
        var = residuals.var()
        return 0.5 * torch.log(2 * math.pi * math.e * var)
    
    def calculate_reward(self, predicted_outcome: torch.Tensor, 
                        actual_outcome: torch.Tensor,
                        experiment_design: torch.Tensor) -> torch.Tensor:
        """
        Multi-scale reward combining accuracy and information gain.
        """
        # Accuracy component (log-likelihood)
        likelihood = -F.mse_loss(predicted_outcome, actual_outcome)
        
        # Information gain component
        info_gain = self.mutual_information(experiment_design, actual_outcome)
        
        # Scaled reward with hyperbolic discounting
        alpha, beta = 0.7, 0.3
        reward = alpha * likelihood + beta * info_gain
        
        return reward

class ScientificLoss(nn.Module):
    """
    Combined loss for SHPM training.
    Includes likelihood, information gain, and calibration components.
    """
    def __init__(self, kl_weight: float = 0.1, entropy_weight: float = 0.05):
        super().__init__()
        self.kl_weight = kl_weight
        self.entropy_weight = entropy_weight
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        losses = {}
        
        # Hypothesis generation loss (cross-entropy)
        if 'hypothesis_logits' in predictions:
            losses['hypothesis'] = F.cross_entropy(
                predictions['hypothesis_logits'].reshape(-1, predictions['hypothesis_logits'].size(-1)),
                targets['hypothesis_ids'].reshape(-1)
            )
        
        # Outcome prediction loss (VAE)
        if 'outcome_dist' in predictions:
            outcome = predictions['outcome_dist']
            # Reconstruction loss
            recon_loss = F.mse_loss(outcome['reconstruction'], targets['outcome'])
            # KL divergence
            kl_loss = -0.5 * torch.sum(
                1 + outcome['log_var'] - outcome['mean'].pow(2) - outcome['log_var'].exp()
            )
            losses['outcome'] = recon_loss + self.kl_weight * kl_loss
        
        # Entropy regularization for exploration
        if 'hypothesis_logits' in predictions:
            probs = F.softmax(predictions['hypothesis_logits'], dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
            losses['entropy'] = -self.entropy_weight * entropy  # Negative to encourage exploration
        
        # Calibration loss (Brier score for uncertainty)
        if 'confidence' in predictions and 'correct' in targets:
            brier_score = torch.mean((predictions['confidence'] - targets['correct'].float())**2)
            losses['calibration'] = brier_score
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses

# %% [markdown]
# ## 4. Reinforcement Learning Environment

# %% Scientific MDP Environment
class ScientificMDP:
    """
    Markov Decision Process for scientific experimentation.
    States: experimental histories
    Actions: hypothesis + experimental design
    Rewards: information gain + prediction accuracy
    """
    def __init__(self, kg_module: KnowledgeGraphModule, 
                 transformer: ScientificTransformer,
                 reward_fn: InformationGainReward):
        self.kg = kg_module
        self.transformer = transformer
        self.reward_fn = reward_fn
        
        # State representation
        self.history = deque(maxlen=100)  # Experimental history
        self.knowledge_state = {}  # Current knowledge graph state
        self.uncertainty_map = {}  # Epistemic uncertainty tracking
        
    def reset(self) -> Dict[str, Any]:
        """Reset environment to initial state"""
        self.history.clear()
        self.knowledge_state = self._initialize_knowledge()
        self.uncertainty_map = defaultdict(lambda: 1.0)  # Max uncertainty
        
        return self.get_state()
    
    def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute experiment and return new state, reward, done flag, and info.
        """
        # Parse action
        hypothesis = action['hypothesis']
        design = action['design']
        
        # Simulate experiment (in real deployment, this would interface with lab)
        outcome = self._simulate_experiment(design)
        
        # Calculate reward
        predicted_outcome = action.get('predicted_outcome', torch.zeros_like(outcome))
        reward = self.reward_fn.calculate_reward(predicted_outcome, outcome, design)
        
        # Update state
        self.history.append({
            'hypothesis': hypothesis,
            'design': design,
            'outcome': outcome,
            'timestamp': len(self.history)
        })
        self._update_knowledge(hypothesis, design, outcome)
        self._update_uncertainty(design, outcome)
        
        # Check termination
        done = self._check_convergence()
        
        # Additional info
        info = {
            'entropy': self._calculate_posterior_entropy(),
            'knowledge_gain': self._measure_knowledge_gain()
        }
        
        return self.get_state(), reward.item(), done, info
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state representation"""
        return {
            'history': list(self.history),
            'knowledge_graph': self.knowledge_state,
            'uncertainty': dict(self.uncertainty_map),
            'embedding': self._encode_state()
        }
    
    def _simulate_experiment(self, design: torch.Tensor) -> torch.Tensor:
        """Simulate experimental outcome (placeholder for real lab interface)"""
        # In production, this would interface with automated lab equipment
        # or query a high-fidelity simulator (e.g., quantum chemistry)
        noise = torch.randn_like(design) * 0.1
        outcome = torch.sigmoid(design + noise)  # Simplified simulation
        return outcome
    
    def _initialize_knowledge(self) -> Dict:
        """Initialize knowledge graph with domain primitives"""
        return {
            'entities': {},
            'relations': {},
            'confidence': 0.1
        }
    
    def _update_knowledge(self, hypothesis: Any, design: Any, outcome: Any):
        """Update knowledge graph with new experimental result"""
        # Extract entities and relations from hypothesis
        # Update graph structure and weights
        pass
    
    def _update_uncertainty(self, design: torch.Tensor, outcome: torch.Tensor):
        """Update epistemic uncertainty map"""
        # Reduce uncertainty in explored regions
        key = tuple(design.flatten().tolist()[:5])  # Hash first 5 dims
        current_uncertainty = self.uncertainty_map[key]
        self.uncertainty_map[key] = current_uncertainty * 0.9  # Decay
    
    def _check_convergence(self) -> bool:
        """Check if discovery process has converged"""
        if len(self.history) < 10:
            return False
        
        # Check if entropy is below threshold
        entropy = self._calculate_posterior_entropy()
        return entropy < 0.1
    
    def _calculate_posterior_entropy(self) -> float:
        """Calculate posterior entropy over hypothesis space"""
        if not self.history:
            return 1.0
        
        # Simplified: use variance of recent outcomes
        recent_outcomes = [h['outcome'] for h in list(self.history)[-10:]]
        if recent_outcomes:
            variance = torch.var(torch.stack(recent_outcomes))
            return float(0.5 * torch.log(2 * math.pi * math.e * variance))
        return 1.0
    
    def _measure_knowledge_gain(self) -> float:
        """Measure knowledge gain from initial state"""
        return 1.0 - self._calculate_posterior_entropy()
    
    def _encode_state(self) -> torch.Tensor:
        """Encode state into vector representation"""
        # Simplified encoding
        if self.history:
            recent = torch.stack([h['outcome'] for h in list(self.history)[-5:]])
            return recent.mean(dim=0)
        return torch.zeros(256)

# %% [markdown]
# ## 5. PPO Training Loop

# %% PPO Agent
class PPOAgent:
    """
    Proximal Policy Optimization for scientific discovery.
    Balances exploration and exploitation with clipped objective.
    """
    def __init__(self, model: ScientificTransformer, kg: KnowledgeGraphModule,
                 lr: float = 1e-4, gamma: float = 0.99, eps_clip: float = 0.2):
        self.model = model
        self.kg = kg
        self.optimizer = optim.AdamW(
            list(model.parameters()) + list(kg.parameters()), 
            lr=lr
        )
        self.gamma = gamma
        self.eps_clip = eps_clip
        
        # Value network for advantage estimation
        self.value_net = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
    def act(self, state: Dict[str, Any]) -> Tuple[Dict[str, Any], torch.Tensor]:
        """Select action using current policy"""
        # Encode state
        state_embedding = state['embedding'].unsqueeze(0)
        
        # Get KG features
        kg_features = self._extract_kg_features(state['knowledge_graph'])
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids=torch.zeros(1, 1, dtype=torch.long),  # Dummy input
                kg_features=kg_features
            )
        
        # Sample hypothesis from distribution
        hypothesis_probs = F.softmax(outputs['hypothesis_logits'][0, -1], dim=-1)
        hypothesis_dist = Categorical(hypothesis_probs)
        hypothesis = hypothesis_dist.sample()
        
        # Get experimental design
        design = outputs['design']
        
        # Predict outcome
        predicted_outcome = outputs['outcome_dist']['mean']
        
        action = {
            'hypothesis': hypothesis,
            'design': design,
            'predicted_outcome': predicted_outcome
        }
        
        # Calculate log probability for PPO
        log_prob = hypothesis_dist.log_prob(hypothesis)
        
        return action, log_prob
    
    def update(self, trajectories: List[Dict]) -> Dict[str, float]:
        """Update policy using PPO objective"""
        losses = []
        
        for trajectory in trajectories:
            states = trajectory['states']
            actions = trajectory['actions']
            rewards = trajectory['rewards']
            log_probs_old = trajectory['log_probs']
            
            # Calculate returns and advantages
            returns = self._calculate_returns(rewards)
            values = torch.cat([self.value_net(s['embedding'].unsqueeze(0)) 
                              for s in states])
            advantages = returns - values.squeeze()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO update
            for _ in range(4):  # PPO epochs
                # Recalculate log probs with current policy
                log_probs_new = []
                for s, a in zip(states, actions):
                    kg_features = self._extract_kg_features(s['knowledge_graph'])
                    outputs = self.model(
                        input_ids=torch.zeros(1, 1, dtype=torch.long),
                        kg_features=kg_features
                    )
                    probs = F.softmax(outputs['hypothesis_logits'][0, -1], dim=-1)
                    dist = Categorical(probs)
                    log_probs_new.append(dist.log_prob(a['hypothesis']))
                
                log_probs_new = torch.stack(log_probs_new)
                
                # Calculate ratio and clipped objective
                ratio = torch.exp(log_probs_new - log_probs_old)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                
                # Policy loss
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), returns)
                
                # Total loss
                total_loss = policy_loss + 0.5 * value_loss
                
                # Backprop
                self.optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                self.value_optimizer.step()
                
                losses.append({
                    'policy': policy_loss.item(),
                    'value': value_loss.item(),
                    'total': total_loss.item()
                })
        
        return {k: np.mean([l[k] for l in losses]) for k in losses[0].keys()}
    
    def _calculate_returns(self, rewards: List[float]) -> torch.Tensor:
        """Calculate discounted returns"""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32)
    
    def _extract_kg_features(self, kg_state: Dict) -> torch.Tensor:
        """Extract features from knowledge graph state"""
        # Simplified: return random features
        return torch.randn(1, 10, 256)

# %% [markdown]
# ## 6. Meta-Learning for Domain Transfer

# %% MAML Implementation
class MAML:
    """
    Model-Agnostic Meta-Learning for few-shot domain adaptation.
    Enables rapid transfer to new scientific domains.
    """
    def __init__(self, model: nn.Module, inner_lr: float = 0.01, 
                 outer_lr: float = 1e-4, inner_steps: int = 5):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.meta_optimizer = optim.Adam(model.parameters(), lr=outer_lr)
        
    def inner_loop(self, task_data: Dict, fast_weights: Dict) -> Tuple[float, Dict]:
        """Inner loop adaptation on single task"""
        support_x, support_y = task_data['support']
        
        # Clone model with fast weights
        if fast_weights is None:
            fast_weights = {name: param.clone() 
                          for name, param in self.model.named_parameters()}
        
        # Inner loop optimization
        for _ in range(self.inner_steps):
            # Forward pass with fast weights
            loss = self._compute_loss(support_x, support_y, fast_weights)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            
            # Update fast weights
            fast_weights = {name: param - self.inner_lr * grad
                          for (name, param), grad in zip(fast_weights.items(), grads)}
        
        # Evaluate on query set
        query_x, query_y = task_data['query']
        query_loss = self._compute_loss(query_x, query_y, fast_weights)
        
        return query_loss, fast_weights
    
    def outer_loop(self, tasks: List[Dict]) -> float:
        """Outer loop meta-optimization"""
        meta_loss = 0
        
        for task in tasks:
            # Inner loop adaptation
            query_loss, _ = self.inner_loop(task, None)
            meta_loss += query_loss
        
        # Meta-gradient update
        meta_loss /= len(tasks)
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def _compute_loss(self, x: torch.Tensor, y: torch.Tensor, 
                     weights: Dict) -> torch.Tensor:
        """Compute loss with given weights"""
        # Simplified: MSE loss
        # In practice, would do full forward pass with weights
        pred = x @ list(weights.values())[0][:x.size(-1)]  # Simplified
        return F.mse_loss(pred, y)

# %% [markdown]
# ## 7. Training Pipeline

# %% Main Training Loop
class SHPMTrainer:
    """
    Complete training pipeline for Scientific Hypothesis Predictor Macro.
    Includes pre-training, fine-tuning, and meta-learning phases.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize modules
        self.kg = KnowledgeGraphModule(
            entity_dim=config['kg_dim'],
            relation_dim=config['relation_dim']
        )
        self.transformer = ScientificTransformer(
            vocab_size=config['vocab_size
            
           ##unfinished
           
           
           
# %% \[markdown]

# ## 7 (continued). Concrete Implementations & Completed Training Pipeline

#

# The cells below provide concrete, runnable (toy-scale) implementations of the remaining

# pieces that were referenced earlier but left as `...`. They are intentionally **simplified**

# so you can run the notebook on a personal machine for development / debugging. Replace

# the toy simulators, small transformer, and random KG features with production-grade

# components (real KG backends, HuggingFace models, domain simulators, and real datasets)

# when scaling up.

# %% Hyperbolic Embeddings (simple Poincaré-ball projection)

class HyperbolicEmbedding(nn.Module):
"""
Toy hyperbolic embedding: stores vectors and clamps them to the Poincaré ball.
Provides a Poincaré distance function and a projector that keeps norms < 1 - eps.
NOTE: This is a lightweight numeric projection, not a full Riemannian optimizer.
"""
def **init**(self, num\_entities: int, dim: int, eps: float = 1e-3):
super().**init**()
self.emb = nn.Embedding(num\_entities, dim)
self.eps = eps
self.dim = dim
\# init small so norms < 1
nn.init.normal\_(self.emb.weight, mean=0.0, std=0.01)

```
def forward(self, idx: torch.LongTensor) -> torch.Tensor:
    z = self.emb(idx)
    # clamp norms to be strictly less than 1 - eps
    norms = z.norm(dim=-1, keepdim=True)
    max_norm = 1.0 - self.eps
    scale = torch.clamp(max_norm / (norms + 1e-9), max=1.0)
    z = z * scale
    return z

@staticmethod
def poincare_distance(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # u, v: (..., dim) in ball with ||·|| < 1
    uu = (u * u).sum(dim=-1)
    vv = (v * v).sum(dim=-1)
    uv = ((u - v) * (u - v)).sum(dim=-1)
    # cosh^-1(1 + 2 * ||u - v||^2 / ((1 - ||u||^2)(1 - ||v||^2)))
    denom = (1.0 - uu) * (1.0 - vv)
    # clamp denom to avoid div-by-zero
    denom = torch.clamp(denom, min=1e-9)
    arg = 1.0 + 2.0 * uv / denom
    # numerical safety
    arg = torch.clamp(arg, min=1.0 + 1e-7)
    return torch.acosh(arg)
```

# %% Knowledge Graph Module (toy GraphSAGE-like)

class KnowledgeGraphModule(nn.Module):
"""
Simple KG module that accepts a graph object and returns node-level pooled embeddings.
For real deployment, swap with a Neo4j-backed pipeline or a production GraphSAGE pipeline.
"""
def **init**(self, entity\_dim: int = 256, relation\_dim: int = 64, hidden: int = 256):
super().**init**()
\# small embedding layers for entities and relations
self.entity\_emb = nn.Linear(entity\_dim, hidden)
self.relation\_emb = nn.Linear(relation\_dim, hidden)
\# a couple of graph conv layers (toy)
self.conv1 = GCNConv(hidden, hidden)
self.conv2 = GCNConv(hidden, hidden)
self.pool = global\_mean\_pool
self.relu = nn.ReLU()

```
def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """
    x: node features (num_nodes, entity_dim)
    edge_index: (2, num_edges)
    batch: (num_nodes,) mapping to graph id
    """
    h = self.relu(self.entity_emb(x))
    h = self.conv1(h, edge_index)
    h = self.relu(h)
    h = self.conv2(h, edge_index)
    # pooled graph embedding
    g = self.pool(h, batch)  # (num_graphs, hidden)
    return g

def random_kg_embedding(self, num_graphs: int = 1) -> torch.Tensor:
    """Utility: returns a random KG embedding for toy runs."""
    return torch.randn(num_graphs, 256)
```

# %% Scientific Transformer (toy transformer encoder + heads)

class ScientificTransformer(nn.Module):
"""
Lightweight transformer-based module that:
\- encodes a (dummy) token sequence
\- cross-attends to KG features (via concatenation here for simplicity)
\- emits:
\* hypothesis\_logits (for discrete hypothesis token generation)
\* design vector (continuous parameterization of experimental design)
\* outcome distribution params (mean, logvar)
Replace with a GPT-like LM + cross-attention to KG for production.
"""
def **init**(self, vocab\_size: int = 4096, d\_model: int = 768, nhead: int = 8, num\_layers: int = 4,
design\_dim: int = 64, outcome\_dim: int = 1, max\_len: int = 32):
super().**init**()
self.token\_emb = nn.Embedding(vocab\_size, d\_model)
encoder\_layer = nn.TransformerEncoderLayer(d\_model=d\_model, nhead=nhead, dim\_feedforward=2048)
self.transformer = nn.TransformerEncoder(encoder\_layer, num\_layers=num\_layers)
self.pos\_emb = nn.Parameter(torch.randn(max\_len, d\_model) \* 0.01)
self.d\_model = d\_model
self.max\_len = max\_len

```
    # heads
    self.hypothesis_head = nn.Linear(d_model, vocab_size)  # for discrete choice / token logits
    self.design_head = nn.Sequential(
        nn.Linear(d_model, d_model // 2),
        nn.ReLU(),
        nn.Linear(d_model // 2, design_dim)
    )
    # outcome distribution head (VAE-like)
    self.outcome_mean = nn.Linear(d_model, outcome_dim)
    self.outcome_logvar = nn.Linear(d_model, outcome_dim)

def forward(self, input_ids: torch.Tensor, kg_features: torch.Tensor = None):
    """
    input_ids: (batch, seq_len) token ids (here often dummy)
    kg_features: (batch, kg_dim) pooled KG embedding
    """
    b, seq_len = input_ids.shape
    # embed tokens and positions
    tok = self.token_emb(input_ids)  # (b, seq_len, d_model)
    pos = self.pos_emb[:seq_len].unsqueeze(0).expand(b, -1, -1)
    h = tok + pos

    # simple fusion: append kg_features as an extra token embedding (projected)
    if kg_features is not None:
        # project KG into d_model
        kg_proj = nn.Linear(kg_features.size(-1), self.d_model).to(kg_features.device)
        kg_token = kg_proj(kg_features).unsqueeze(1)  # (b,1,d_model)
        # concat and run transformer
        h = torch.cat([h, kg_token], dim=1)  # seq_len + 1
    # transformer expects (seq_len, batch, d_model)
    h_t = h.permute(1, 0, 2)
    h_out = self.transformer(h_t)  # (seq_len(+1), b, d_model)
    h_out = h_out.permute(1, 0, 2)  # (b, seq_len(+1), d_model)

    # take last token as summary (if kg appended, last is kg token; else last token)
    summary = h_out[:, -1, :]  # (b, d_model)

    hypothesis_logits = self.hypothesis_head(h_out)  # logits per position
    design_vector = self.design_head(summary)  # continuous design vector
    mean = self.outcome_mean(summary)
    logvar = self.outcome_logvar(summary)

    return {
        'hypothesis_logits': hypothesis_logits,  # (b, seq_len(+1), vocab_size)
        'design': design_vector,                 # (b, design_dim)
        'outcome_dist': {'mean': mean, 'logvar': logvar}
    }
```

# %% Information-Theoretic Reward

class InformationGainReward:
"""
Compute a reward combining log-likelihood and (estimated) information gain.
For the toy example we treat predicted outcome as Gaussian (mean, var).
"""
def **init**(self, alpha: float = 0.7, beta: float = 0.3, min\_var: float = 1e-3):
self.alpha = alpha
self.beta = beta
self.min\_var = min\_var

```
def gaussian_log_likelihood(self, mean: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    var = torch.exp(logvar) + self.min_var
    log_prob = -0.5 * (math.log(2 * math.pi) + torch.log(var) + ((y - mean) ** 2) / var)
    return log_prob.sum(dim=-1)

def estimate_ig(self, prior_logvar: torch.Tensor, posterior_logvar: torch.Tensor) -> torch.Tensor:
    """
    Information gain approx as reduction in entropy for Gaussian:
    H = 0.5 * log(2*pi*e*var) -> IG = H_prior - H_post = 0.5 * log(var_prior / var_post)
    """
    var_prior = torch.exp(prior_logvar) + self.min_var
    var_post = torch.exp(posterior_logvar) + self.min_var
    ig = 0.5 * torch.log(var_prior / var_post)
    # sum across outcome dims
    return ig.sum(dim=-1)

def __call__(self, pred_mean: torch.Tensor, pred_logvar: torch.Tensor,
             prior_logvar: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    ll = self.gaussian_log_likelihood(pred_mean, pred_logvar, observed)  # (batch,)
    ig = self.estimate_ig(prior_logvar, pred_logvar)  # (batch,)
    reward = self.alpha * ll + self.beta * ig
    return reward
```

# %% Scientific MDP (toy environment)

class ScientificMDP:
"""
Toy environment capturing the 'experiment' loop.
State: history (simplified), KG state
Action: dict with 'hypothesis' (int token), 'design' vector
The environment simulates an outcome using a simple parametric function.
"""
def **init**(self, kg\_module: KnowledgeGraphModule, outcome\_noise: float = 0.1, seed: int = 42):
self.kg = kg\_module
self.outcome\_noise = outcome\_noise
np.random.seed(seed)
self.reset()

```
def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
    # Clear history and produce a random starting KG
    self.history = []
    # Toy KG features: a random graph per episode
    self.num_nodes = np.random.randint(4, 12)
    node_feats = torch.randn(self.num_nodes, 256)
    # random edge_index forming an undirected Erdos-Renyi graph
    edges = []
    for i in range(self.num_nodes):
        for j in range(i + 1, self.num_nodes):
            if np.random.rand() < 0.2:
                edges.append([i, j])
                edges.append([j, i])
    if len(edges) == 0:
        # ensure at least one edge
        edges = [[0, 1], [1, 0]]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    batch = torch.zeros(self.num_nodes, dtype=torch.long)
    self.kg_state = {'node_feats': node_feats, 'edge_index': edge_index, 'batch': batch}
    # state embedding: random for the toy example
    state_embedding = torch.randn(768)
    self.current_state = {
        'embedding': state_embedding,
        'knowledge_graph': self.kg_state
    }
    self.t = 0
    return self.current_state

def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict]:
    """
    Run an experiment given `action` and return (next_state, reward, done, info).
    The simulator maps `design` -> outcome via a toy nonlinear function plus noise.
    """
    design = action['design']  # tensor (design_dim,)
    # simple deterministic function: dot with random target + nonlinearity
    target = torch.sin(torch.linspace(0.1, 1.0, design.size(-1)))
    deterministic = (design.squeeze() * target).sum()
    noise = torch.randn_like(deterministic) * self.outcome_noise
    observed = deterministic + noise
    # assemble tensors in consistent shapes
    pred_mean = action['predicted_outcome'].squeeze() if isinstance(action['predicted_outcome'], torch.Tensor) else torch.tensor(deterministic)
    pred_logvar = torch.zeros_like(pred_mean) - 1.0  # toy confident prior
    # reward will be computed externally (agent supplies reward handler)
    self.history.append({'action': action, 'observed': observed.item()})
    self.t += 1
    # update state (toy)
    self.current_state['embedding'] = torch.randn_like(self.current_state['embedding'])
    done = self.t >= 5  # short episodes in toy env
    info = {'observed': observed}
    return self.current_state, torch.tensor([observed]), done, info
```

# %% Utilities: checkpointing and eval

def save\_checkpoint(path: str, model: nn.Module, kg: KnowledgeGraphModule, optimizer: optim.Optimizer, step: int):
state = {
'model\_state': model.state\_dict(),
'kg\_state': kg.state\_dict(),
'optimizer\_state': optimizer.state\_dict(),
'step': step
}
torch.save(state, path)
logger.info(f"Saved checkpoint to {path}")

def load\_checkpoint(path: str, model: nn.Module, kg: KnowledgeGraphModule, optimizer: optim.Optimizer):
state = torch.load(path, map\_location='cpu')
model.load\_state\_dict(state\['model\_state'])
kg.load\_state\_dict(state\['kg\_state'])
optimizer.load\_state\_dict(state\['optimizer\_state'])
logger.info(f"Loaded checkpoint from {path}")
return state\['step']

# %% Finish SHPMTrainer class

class SHPMTrainer:
"""
Completed trainer combining:
\- small pretraining placeholder,
\- PPO-based fine-tuning with the PPOAgent above,
\- small MAML meta-loop.
This is a DEVELOPMENT helper to test the end-to-end loop.
"""
def **init**(self, config: Dict\[str, Any]):
self.config = config

```
    # initialize KG and transformer (toy)
    self.kg = KnowledgeGraphModule(entity_dim=config.get('kg_dim', 256),
                                   relation_dim=config.get('relation_dim', 64),
                                   hidden=config.get('kg_hidden', 256))
    self.transformer = ScientificTransformer(vocab_size=config.get('vocab_size', 4096),
                                             d_model=config.get('d_model', 768),
                                             nhead=config.get('nhead', 8),
                                             num_layers=config.get('num_layers', 3),
                                             design_dim=config.get('design_dim', 32),
                                             outcome_dim=config.get('outcome_dim', 1),
                                             max_len=config.get('max_len', 16))
    # PPO agent
    self.agent = PPOAgent(self.transformer, self.kg,
                          lr=config.get('lr', 1e-4),
                          gamma=config.get('gamma', 0.99),
                          eps_clip=config.get('eps_clip', 0.2))
    # MAML (toy)
    self.maml = MAML(self.transformer, inner_lr=config.get('maml_inner_lr', 1e-2),
                     outer_lr=config.get('maml_outer_lr', 1e-4),
                     inner_steps=config.get('maml_inner_steps', 3))
    # reward function
    self.reward_fn = InformationGainReward(alpha=config.get('alpha', 0.7),
                                           beta=config.get('beta', 0.3))
    # environment
    self.env = ScientificMDP(self.kg, outcome_noise=config.get('outcome_noise', 0.1))
    # misc
    self.device = config.get('device', 'cpu')
    self.checkpoint_path = config.get('checkpoint_path', 'shpm_ckpt.pt')
    # move modules to device
    self.transformer.to(self.device)
    # small replay buffer
    self.replay = deque(maxlen=config.get('replay_size', 1000))

def pretrain_placeholder(self, steps: int = 100):
    """
    Placeholder pretraining: run a few steps of masked LM on dummy data so weights are non-random.
    Replace this with real pretraining on corpora (arXiv/PubMed/ChEMBL) in production.
    """
    logger.info("Starting placeholder pretraining...")
    optim_p = optim.Adam(self.transformer.parameters(), lr=1e-4)
    for step in range(steps):
        # dummy batch
        input_ids = torch.randint(0, self.transformer.hypothesis_head.out_features, (4, 8)).to(self.device)
        outputs = self.transformer(input_ids, kg_features=torch.randn(4, 256).to(self.device))
        logits = outputs['hypothesis_logits'][:, :8, :]
        target = torch.randint(0, logits.size(-1), (4, 8)).to(self.device)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        optim_p.zero_grad()
        loss.backward()
        optim_p.step()
        if step % 20 == 0:
            logger.info(f"Pretrain step {step}/{steps} loss={loss.item():.4f}")
    logger.info("Finished placeholder pretraining.")

def generate_trajectory(self, max_steps: int = 5) -> Dict:
    """Run one interaction episode and collect trajectory (toy)."""
    state = self.env.reset()
    traj = {'states': [], 'actions': [], 'rewards': [], 'log_probs': []}
    done = False
    while not done:
        # agent.act expects state with embedding and knowledge_graph
        action, log_prob = self.agent.act(state)
        next_state, observed, done, info = self.env.step(action)
        # compute reward using reward function (toy): predicted mean==action['predicted_outcome']
        # NOTE: here pred_logvar and prior_logvar are placeholder scalars
        pred_mean = action['predicted_outcome'] if isinstance(action['predicted_outcome'], torch.Tensor) else torch.tensor([action['predicted_outcome']])
        pred_logvar = torch.zeros_like(pred_mean) - 1.0
        prior_logvar = torch.zeros_like(pred_mean) - 0.5  # prior less confident
        reward = self.reward_fn(pred_mean, pred_logvar, prior_logvar, observed)
        # store
        traj['states'].append(state)
        traj['actions'].append(action)
        traj['rewards'].append(reward.item() if isinstance(reward, torch.Tensor) else float(reward))
        traj['log_probs'].append(log_prob)
        state = next_state
    return traj

def fine_tune_ppo(self, epochs: int = 10, episodes_per_epoch: int = 8):
    """
    Fine-tune with PPO on toy env.
    Replace with large-scale RL over real simulator / experimental database for production.
    """
    logger.info("Starting PPO fine-tuning...")
    for ep in range(epochs):
        trajectories = []
        for _ in range(episodes_per_epoch):
            traj = self.generate_trajectory()
            trajectories.append(traj)
            # add to replay
            self.replay.append(traj)
        metrics = self.agent.update(trajectories)
        logger.info(f"PPO epoch {ep}/{epochs}: policy_loss={metrics['policy']:.4f} value_loss={metrics['value']:.4f} total_loss={metrics['total']:.4f}")
        # checkpoint occasionally
        if ep % max(1, epochs // 5) == 0:
            save_checkpoint(self.checkpoint_path, self.transformer, self.kg, self.agent.optimizer, step=ep)
    logger.info("Finished PPO fine-tuning.")

def meta_train(self, tasks: List[Dict], meta_epochs: int = 5):
    """
    Toy meta-training using the MAML wrapper.
    Each task in `tasks` should contain 'support' and 'query' sets.
    For demo we create synthetic tasks.
    """
    logger.info("Starting meta-training...")
    for epoch in range(meta_epochs):
        meta_loss = self.maml.outer_loop(tasks)
        logger.info(f"Meta epoch {epoch}/{meta_epochs}: meta_loss={meta_loss:.4f}")
    logger.info("Finished meta-training.")

def evaluate(self, num_episodes: int = 20) -> Dict[str, float]:
    """Run evaluation episodes and report mean reward and simple calibration metrics."""
    rewards = []
    for _ in range(num_episodes):
        traj = self.generate_trajectory()
        rewards.append(sum(traj['rewards']))
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    logger.info(f"Evaluation: mean_reward={mean_reward:.4f} std={std_reward:.4f}")
    return {'mean_reward': mean_reward, 'std_reward': std_reward}

def train(self, pretrain_steps: int = 100, ppo_epochs: int = 10):
    """
    Full pipeline: pretrain (placeholder) -> PPO fine-tune -> evaluate -> meta-train (toy)
    """
    self.pretrain_placeholder(steps=pretrain_steps)
    self.fine_tune_ppo(epochs=ppo_epochs, episodes_per_epoch=self.config.get('episodes_per_epoch', 8))
    eval_stats = self.evaluate(num_episodes=self.config.get('eval_episodes', 16))
    # Toy meta tasks (synth)
    tasks = []
    for _ in range(4):
        # create tiny synthetic support/query (random tensors)
        support_x = torch.randn(5, 16)
        support_y = torch.randn(5, 1)
        query_x = torch.randn(5, 16)
        query_y = torch.randn(5, 1)
        tasks.append({'support': (support_x, support_y), 'query': (query_x, query_y)})
    self.meta_train(tasks, meta_epochs=2)
    # final checkpoint
    save_checkpoint(self.checkpoint_path, self.transformer, self.kg, self.agent.optimizer, step=999)
    return eval_stats
```

# %% \[markdown]

# ## 8. Example usage (toy demo)

# The example below runs the toy pipeline end-to-end. On a laptop it will be quick; on larger hardware

# you can increase `pretrain_steps` / `ppo_epochs` to exercise heavier workloads.

if **name** == "**main**":
\# Example config (toy)
config = {
'kg\_dim': 256,
'relation\_dim': 64,
'kg\_hidden': 256,
'vocab\_size': 2048,
'd\_model': 256,
'nhead': 8,
'num\_layers': 2,
'design\_dim': 32,
'outcome\_dim': 1,
'max\_len': 12,
'lr': 3e-4,
'gamma': 0.99,
'eps\_clip': 0.2,
'maml\_inner\_lr': 1e-2,
'maml\_outer\_lr': 1e-4,
'maml\_inner\_steps': 3,
'outcome\_noise': 0.05,
'replay\_size': 500,
'checkpoint\_path': 'shpm\_toy\_ckpt.pt',
'device': 'cpu',
'episodes\_per\_epoch': 4,
'eval\_episodes': 8
}

```
trainer = SHPMTrainer(config)
stats = trainer.train(pretrain_steps=50, ppo_epochs=6)
print("Toy training complete. Eval stats:", stats)
```

# %% \[markdown]

# ## 9. Notes for productionization

# - Replace toy transformer with a pretrained LM (SciBERT / Galactica / domain-tuned GPT) and add cross-attention to real KG embeddings.

# - Replace `ScientificMDP` simulator with domain-specific lab simulators or an interface to ELNs/robotic platforms.

# - Use a production KG backend (Neo4j, Amazon Neptune) and an index/store for primitives (vector DB + metadata).

# - Implement rigorous uncertainty quantification: ensemble posteriors, MCMC, or Bayesian last-layer approaches rather than toy Gaussian heads.

# - Add dataset ingestion pipelines: PUBCHEM, ChEMBL, PubMed, Materials Project, ReDO (negative result corpora) and robust parsers for methods/protocol extraction.

# - Strengthen safety: human-in-the-loop gating for high-risk experiments and ethical oversight for proposals that could cause harm.

#

# This appended section completes the notebook skeleton with executable toy pieces so you can iterate quickly while developing the full-scale SHPM pipeline.

