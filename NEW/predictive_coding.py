import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class PredictiveCodingLayer(nn.Module):
    """
    A predictive coding layer that implements energy minimization through iterative relaxation.
    Each neuron tries to minimize the prediction error between its activation and the 
    prediction from the previous layer.
    """
    def __init__(self, in_dim, out_dim, relax_steps=5, alpha=0.1, eta=1e-3):
        """
        Args:
            in_dim: Input dimension
            out_dim: Output dimension  
            relax_steps: Number of relaxation steps for energy minimization
            alpha: Learning rate for neuron state updates
            eta: Learning rate for weight updates
        """
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)
        self.relax_steps = relax_steps
        self.alpha = alpha  # step size for neuron state update
        self.eta = eta      # local learning rate

    def forward(self, x_in):
        # Initialize layer activations
        x = torch.zeros(x_in.size(0), self.W.size(0), device=x_in.device, requires_grad=False)

        # Iterative relaxation to minimize prediction error
        for _ in range(self.relax_steps):
            mu = F.linear(x_in, self.W)        # prediction from lower layer
            eps = x - mu                       # prediction error
            # Update neuron activations to reduce energy
            x = x - self.alpha * eps

        # Optional local Hebbian-like update during inference
        with torch.no_grad():
            eps = x - F.linear(x_in, self.W)
            # Compute weight updates using outer product per batch
            dW = torch.bmm(eps.unsqueeze(2), x_in.unsqueeze(1))  # [batch, out_dim, 1] @ [batch, 1, in_dim]
            self.W += self.eta * dW.mean(0)

        return x


class PredictiveCodingAutoencoder(nn.Module):
    """
    An autoencoder using predictive coding layers in both encoder and decoder.
    """
    def __init__(self, input_dim=784, hidden_dims=[256, 64], relax_steps=5):
        """
        Args:
            input_dim: Dimension of input data
            hidden_dims: List of hidden layer dimensions from input to latent space
            relax_steps: Number of relaxation steps for each predictive coding layer
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.relax_steps = relax_steps
        
        # Create encoder layers
        self.encoder_layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            self.encoder_layers.append(
                PredictiveCodingLayer(dims[i], dims[i+1], relax_steps=relax_steps)
            )

        # Create decoder layers (mirror of encoder)
        self.decoder_layers = nn.ModuleList()
        rev_dims = list(reversed(dims))  # Reversed dimensions for decoder
        for i in range(len(rev_dims) - 1):
            self.decoder_layers.append(
                PredictiveCodingLayer(rev_dims[i], rev_dims[i+1], relax_steps=relax_steps)
            )

    def encode(self, x):
        """Encode input to latent representation."""
        for layer in self.encoder_layers:
            x = layer(x)
        return x

    def decode(self, z):
        """Decode latent representation to reconstruction."""
        for layer in self.decoder_layers:
            z = layer(z)
        return z

    def forward(self, x):
        """Forward pass: encode then decode."""
        z = self.encode(x)
        recon = self.decode(z)
        return recon


# MNIST Dataset and DataLoader setup
def get_mnist_dataloader(batch_size=128, flatten=True):
    """
    Set up MNIST dataset and dataloader for the autoencoder.
    
    Args:
        batch_size: Size of training batches
        flatten: Whether to flatten 28x28 images to 784-dimensional vectors
    
    Returns:
        DataLoader for MNIST dataset
    """
    transform_list = [transforms.ToTensor()]
    if flatten:
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))  # flatten 28x28 -> 784
    
    transform = transforms.Compose(transform_list)
    
    train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    return train_loader


def train_predictive_coding_ae(model, train_loader, epochs=5, lr=1e-3, device='cpu'):
    """
    Train the predictive coding autoencoder.
    
    Args:
        model: The predictive coding autoencoder model
        train_loader: DataLoader for training data
        epochs: Number of training epochs
        lr: Learning rate for the optimizer
        device: Device to run training on (cpu or cuda)
    
    Returns:
        List of average losses per epoch
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)
    
    loss_history = []
    
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            
            recon = model(x)
            loss = F.mse_loss(recon, x)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            
            # Optional: print progress
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}, Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / len(train_loader.dataset)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")
    
    return loss_history


def visualize_reconstructions(model, test_loader, device='cpu', num_samples=8):
    """
    Visualize original images and their reconstructions.
    
    Args:
        model: Trained predictive coding autoencoder model
        test_loader: DataLoader for test data
        device: Device to run inference on
        num_samples: Number of samples to visualize
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return
    
    model.eval()
    
    with torch.no_grad():
        # Get a batch of test images
        x_test, _ = next(iter(test_loader))
        x_test = x_test.to(device)
        
        # Select a subset to visualize
        x_subset = x_test[:num_samples]
        recon = model(x_subset)
        
        # Reshape for visualization (assuming MNIST format)
        x_subset = x_subset.view(-1, 1, 28, 28).cpu()
        recon = recon.view(-1, 1, 28, 28).cpu()
        
        fig, axes = plt.subplots(2, num_samples, figsize=(12, 3))
        for i in range(num_samples):
            # Original image
            axes[0, i].imshow(x_subset[i][0], cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original')
            
            # Reconstructed image
            axes[1, i].imshow(recon[i][0], cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstruction')
        
        plt.suptitle('Top: Original | Bottom: Reconstruction')
        plt.tight_layout()
        plt.show()


def main():
    """
    Main function to run the predictive coding autoencoder implementation.
    """
    # Check for GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Set up data
    print("Setting up MNIST dataset...")
    train_loader = get_mnist_dataloader(batch_size=128)
    
    # Create model
    print("Creating predictive coding autoencoder...")
    model = PredictiveCodingAutoencoder(
        input_dim=784,
        hidden_dims=[256, 64],
        relax_steps=5
    )
    
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train model
    print("Starting training...")
    loss_history = train_predictive_coding_ae(
        model=model,
        train_loader=train_loader,
        epochs=5,
        lr=1e-3,
        device=device
    )
    
    # Set up test data loader for visualization
    test_ds = datasets.MNIST(root='./data', train=False, download=True, 
                             transform=transforms.Compose([transforms.ToTensor(), 
                                                          transforms.Lambda(lambda x: x.view(-1))]))
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    # Visualize results
    print("Visualizing reconstructions...")
    visualize_reconstructions(model, test_loader, device=device)
    
    # Plot training loss
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(loss_history)
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.grid(True)
        plt.show()
    except ImportError:
        print("matplotlib not available, skipping loss plot")


if __name__ == "__main__":
    main()