```python
# Cell 1: Imports and Setup
# This notebook prototypes a hierarchical transformer-based model for natural language to command chain generation.
# We use PyTorch for the core architecture, leveraging a pre-trained CodeLlama tokenizer and base model for transfer learning.
# Note: This is a simplified prototype. Full implementation requires significant compute (e.g., multiple GPUs).

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from collections import defaultdict
import random

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pre-trained model and tokenizer (using a smaller variant for demo; scale to CodeLlama-7B for production)
model_name = "microsoft/CodeT5-base"  # Placeholder; replace with "codellama/CodeLlama-7b-hf" for better code focus
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Extend tokenizer if needed for shell commands
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

```python
# Cell 2: Dataset Preparation
# We create a synthetic dataset for demo purposes: pairs of (task_description, command_chain).
# In practice, source from StackOverflow, GitHub, man pages, etc.
# For now, hardcode a small set and augment synthetically.

class CommandDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        task, command = self.data[idx]
        # Format as prompt: "Task: {task}\nCommand: {command}"
        text = f"Task: {task}\nCommand: {command}<EOS>"
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        # Labels: shift input_ids for causal LM
        input_ids = encoding['input_ids'].squeeze()
        labels = input_ids.clone()
        # Mask task part as -100 (ignore loss)
        task_tokens = self.tokenizer(f"Task: {task}\nCommand: ", return_tensors='pt')['input_ids'].squeeze()
        labels[:len(task_tokens)] = -100
        return {
            'input_ids': input_ids.to(device),
            'attention_mask': encoding['attention_mask'].squeeze().to(device),
            'labels': labels.to(device)
        }

# Synthetic data generation (expand with paraphrasing, etc.)
def generate_synthetic_data(num_samples=1000):
    base_tasks = [
        ("List all files in current directory", "ls -la"),
        ("Change to home directory", "cd ~"),
        ("Create a new file named test.txt", "touch test.txt"),
        ("Search for 'error' in log file", "grep 'error' /var/log/syslog | sort | uniq"),
        ("Install package via npm", "npm install lodash"),
        ("Run Python script with args", "python script.py --input data.csv"),
        # Add more: complex chains, etc.
    ]
    data = []
    for _ in range(num_samples // len(base_tasks)):
        for task, cmd in base_tasks:
            # Paraphrase task
            paraphrases = [
                task,
                f"Please {task.lower()}",
                f"How to {task.split(' ',1)[1] if len(task.split()) > 1 else task}?",
                # Add noise: typos, etc.
                re.sub(r'\b(\w+)e\b', r'\1', task)  # Simple variation
            ]
            for para in paraphrases[:2]:  # Limit for demo
                data.append((para.strip(), cmd))
    random.shuffle(data)
    return data[:num_samples]

train_data = generate_synthetic_data(500)
val_data = generate_synthetic_data(100)

train_dataset = CommandDataset(train_data, tokenizer)
val_dataset = CommandDataset(val_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
```

```python
# Cell 3: Hierarchical Intent Decomposition (Simplified)
# We add a lightweight hierarchy on top of the base model:
# - Intent Classifier (MLP on pooled embeddings)
# - Task Decomposer (seq2seq for sub-tasks)
# For prototype, we integrate into the causal LM with prompts; full hier. needs separate modules.

class HierarchicalCommandModel(nn.Module):
    def __init__(self, base_model, num_intents=10, hidden_dim=768):
        super().__init__()
        self.base_model = base_model
        self.intent_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_intents)
        )
        # Decomposer: simple linear for sub-task tokens (prototype)
        self.decomposer = nn.Linear(hidden_dim, hidden_dim)
        self.sequence_optimizer = nn.Linear(hidden_dim, tokenizer.vocab_size)  # For command tokens
    
    def forward(self, input_ids, attention_mask, labels=None):
        # Base forward
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        loss = outputs.loss if labels is not None else None
        
        # Hierarchical: extract embeddings for intent
        hidden_states = self.base_model.transformer.encoder.last_hidden_state  # Assuming encoder-decoder
        pooled = (hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        intent_logits = self.intent_classifier(pooled)
        
        # Decompose: apply to hidden states
        decomposed = torch.relu(self.decomposer(hidden_states))
        seq_logits = self.sequence_optimizer(decomposed)
        
        return {
            'loss': loss,
            'logits': logits,  # Main output
            'intent_logits': intent_logits,
            'seq_logits': seq_logits
        }

# Instantiate model
model = HierarchicalCommandModel(base_model, num_intents=5)  # e.g., file, network, install, etc.
model.train()
```

```python
# Cell 4: Reward Function for RL (Shaped Rewards)
# For Phase 2: Define multi-dimensional reward.
# In prototype, simulate execution in a mock env; real: use sandbox.

def compute_reward(pred_command, true_command, task, mock_env=None):
    # Simulate scores (in real: execute in Docker, check state)
    task_completion = 1.0 if pred_command.strip() == true_command.strip() else 0.5  # Partial
    efficiency = 1.0 / max(len(pred_command.split()), 1)  # Inverse length
    safety = 1.0 if 'rm -rf' not in pred_command else -1.0  # Simple check
    generalization = 0.8  # Mock: similarity via token overlap
    novelty_penalty = -0.1 if random.random() > 0.9 else 0.0  # Rare penalty
    
    alphas = [0.4, 0.2, 0.2, 0.1, 0.1]
    reward = sum(a * s for a, s in zip(alphas, [task_completion, efficiency, safety, generalization, novelty_penalty]))
    return reward

# Example
print(compute_reward("ls -la", "ls -la", "List files"))
```

```python
# Cell 5: Training Loop - Phase 1: Supervised
# Use AdamW with scheduler. Add RL later via PPO or similar.

optimizer = optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=len(train_loader)*num_epochs)

def train_step(batch):
    optimizer.zero_grad()
    outputs = model(**batch)
    loss = outputs['loss']
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item()

losses = []
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch in train_loader:
        loss = train_step(batch)
        epoch_loss += loss
    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}: Loss {avg_loss:.4f}")

# Plot losses
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
```

```python
# Cell 6: Evaluation and Inference
# Generate command from task.

def generate_command(model, task, max_new_tokens=50, temperature=0.8):
    prompt = f"Task: {task}\nCommand:"
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.base_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    command = generated.split("Command:")[-1].strip().split("<EOS>")[0].strip()
    return command

# Test
test_task = "List all files in current directory"
pred_cmd = generate_command(model, test_task)
print(f"Task: {test_task}\nPredicted: {pred_cmd}")
reward = compute_reward(pred_cmd, "ls -la", test_task)
print(f"Reward: {reward:.2f}")
```

```python
# Cell 7: Memory-Augmented Component (Simple DNC Prototype)
# Implement a basic external memory for storing/retrieving command templates.
# Use key-value store with attention.

class SimpleMemory(nn.Module):
    def __init__(self, memory_size=100, embed_dim=768):
        super().__init__()
        self.memory_keys = nn.Parameter(torch.randn(memory_size, embed_dim))
        self.memory_values = nn.Parameter(torch.randn(memory_size, embed_dim))  # Embed commands
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8)
    
    def forward(self, query):  # query: task embedding
        attn_output, _ = self.attention(query.unsqueeze(0), self.memory_keys.unsqueeze(0), self.memory_values.unsqueeze(0))
        return attn_output.squeeze(0)

# Integrate into model (add to forward)
memory = SimpleMemory().to(device)

# Example usage: store a template
task_embed = torch.randn(1, 768).to(device)  # From intent encoder
retrieved = memory(task_embed)
print("Retrieved memory shape:", retrieved.shape)
```

```python
# Cell 8: RL Integration Sketch (PPO-like)
# For Phase 2: Use torchrl or simple policy gradient.
# Prototype: REINFORCE with baseline.

from torch.distributions import Categorical

class RLTrainer:
    def __init__(self, model, gamma=0.99):
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    def compute_returns(self, rewards):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns
    
    def update(self, log_probs, returns):
        returns = torch.tensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        policy_loss = []
        for log_prob, ret in zip(log_probs, returns):
            policy_loss.append(-log_prob * ret)
        loss = torch.stack(policy_loss).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

# Example training step (mock trajectory)
rl_trainer = RLTrainer(model)
log_probs = [torch.tensor(0.5).to(device)] * 5  # From sampling
rewards = [compute_reward("ls", "ls", "list") for _ in range(5)]
returns = rl_trainer.compute_returns(rewards)
rl_loss = rl_trainer.update(log_probs, returns)
print(f"RL Loss: {rl_loss:.4f}")
```

```python
# Cell 9: Curriculum Learning Sketch
# Progressive stages: filter data by complexity.

def get_curriculum_data(stage=1):
    complexities = {
        1: [("ls", "ls")],  # Atomic
        2: [("grep error log", "grep error log.txt")],  # Pipes
        # Add more
    }
    return generate_synthetic_data(100)  # Filtered in full impl.

# Train on stage 1
curr_data = get_curriculum_data(1)
curr_dataset = CommandDataset(curr_data, tokenizer)
# Re-run training loop...
```

```python
# Cell 10: Adversarial Training Sketch
# Train a discriminator on failure prediction.

class FailurePredictor(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),  # Concat task + cmd embed
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, task_embed, cmd_embed):
        combined = torch.cat([task_embed, cmd_embed], dim=-1)
        return self.classifier(combined)

discriminator = FailurePredictor().to(device)

# GAN-like update: main model maximizes success prob, disc minimizes error
# (Implement in loop)
```

```python
# Cell 11: Self-Improvement Loop (Human-in-the-Loop Mock)
# Simulate feedback and update.

def human_feedback(pred, true):
    # Mock: 80% acceptance, else correction
    if random.random() < 0.8:
        correction = true
    else:
        correction = pred.replace("ls", "dir")  # Mock edit
    return correction

# Loop
feedbacks = []
for task, true_cmd in val_data[:5]:
    pred = generate_command(model, task)
    corr = human_feedback(pred, true_cmd)
    feedbacks.append((task, pred, corr))
    # Update: fine-tune on correction
    # (Add to train_data and retrain mini-batch)

print("Sample Feedbacks:")
for t, p, c in feedbacks:
    print(f"Task: {t}\nPred: {p}\nCorr: {c}\n")
```

```python
# Cell 12: Emergence Module - Graph Attention for Composition
# Use NetworkX + GAT for command dependency graph.

import networkx as nx
from torch_geometric.nn import GATConv  # Requires torch_geometric (assume available or install)

class CompositionGAT(nn.Module):
    def __init__(self, in_channels=768, out_channels=768):
        super().__init__()
        self.gat = GATConv(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        return self.gat(x, edge_index)

# Mock graph: nodes=commands, edges=dependencies
G = nx.DiGraph()
G.add_edges_from([("ls", "cd"), ("grep", "ls")])  # ls before grep, etc.
# Convert to PyG format...
# x = node embeds
# edge_index = torch.tensor(G.edges)

# In full: learn compositions for novel chains
```

This notebook provides a practical, feasible prototype implementing key elements of the described architecture:
- **Base**: Pre-trained CodeT5 for transfer learning.
- **Hierarchy**: Simplified intent/decomposition layers.
- **Memory**: Basic attention-based external memory.
- **Training Phases**: Supervised + RL sketch with shaped rewards.
- **Curriculum**: Staged data complexity.
- **Adversarial**: Failure predictor outline.
- **Feedback**: Human-in-loop simulation.
- **Emergence**: GAT for compositional reasoning.

For production:
- Scale to larger models (e.g., CodeLlama-7B).
- Integrate real sandbox (Docker API for execution).
- Use RL libraries like Stable Baselines3 for full PPO.
- Data: Scrape SO/GitHub with ethical sourcing.
- Compute: Distributed training via torch.distributed.

Philosophically, this design balances **exploration** (novelty in curriculum/RL) with **exploitation** (memory/retrieval), driving toward **emergence** via compositionality and meta-learning. Mathematically, the multi-objective reward optimizes a Pareto front via weighted sum, with gradient descent ensuring efficient convergence (e.g., via cosine LR decay). Intervals: Daily data ingestion, weekly RL updates, monthly evals—ensuring multiplicative improvement through feedback loops.
