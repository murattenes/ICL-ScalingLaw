"""
Training script for Task 5 (v2): Non-Linear Extension with 2-Layer NN.

Compares Linear Model vs 2-Layer Neural Network on Non-Linear Isotropic Data.

Based on TA's suggestion:
- Model 1 (Linear): f̂ = w^T · h^L  (linear baseline)
- Model 2 (2-Layer NN): f̂ = a^T · g(W_0 · h^L)  where g = ReLU

where h^L is the D-dimensional representation combining x* and context info.

Data:
- y = ReLU(β·x / √D) + σε  (single-index nonlinear target)

Key difference from ScalarHeadModel:
- ScalarHeadModel applies 1D nonlinearity to scalar prediction u
- TwoLayerNNModel applies proper 2-layer NN to D-dimensional representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Tuple
import os
import sys
import math

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import ReducedGammaModel, compute_icl_loss


class LinearModelWithBias(nn.Module):
    """
    Linear Model (Model 1 from TA's note): f̂ = w^T · h^L + bias

    This is equivalent to the paper's reduced gamma model with an added bias.
    The representation h^L = x* ⊙ rep combines test input with context info.

    Architecture:
        rep = γ * S @ X^T @ y   (D-dimensional, learned from context)
        f(x*) = (x* · rep) / (LP) + bias
    """

    def __init__(self, D: int, L: int, init_gamma: float = 0.0):
        super().__init__()
        self.D = D
        self.L = L
        self.gamma = nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, X: torch.Tensor, y: torch.Tensor, X_star: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for linear model.

        Args:
            X: Training inputs (P, D)
            y: Training targets (P,)
            X_star: Test inputs (K, D)

        Returns:
            Predictions (K,)
        """
        P, D = X.shape
        L = self.L
        y = y.squeeze()

        # Compute empirical covariance: Σ̂ = (1/P) X^T X
        Sigma_hat = (X.T @ X) / P  # (D, D)

        # Compute M = I - (γ/L) Σ̂
        I = torch.eye(D, device=X.device, dtype=X.dtype)
        M = I - (self.gamma / L) * Sigma_hat

        # Compute geometric series: S = Σ_{ℓ=0}^{L-1} M^ℓ
        S = torch.zeros_like(I)
        M_power = I.clone()
        for ell in range(L):
            S = S + M_power
            if ell < L - 1:
                M_power = M_power @ M

        # Representation: rep = γ * S @ X^T @ y  (D-dimensional)
        rep = self.gamma * (S @ (X.T @ y))

        # Linear prediction: f(x*) = (x* · rep) / (LP) + bias
        predictions = (X_star @ rep) / (L * P) + self.bias

        return predictions


class TwoLayerNNModel(nn.Module):
    """
    2-Layer Neural Network Model (Model 2 from TA's note): f̂ = a^T · g(W_0 · h^L) + b

    where:
        h^L = x* ⊙ rep / (LP)  (D-dimensional, combines x* with context)
        W_0 ∈ R^{k×D}  (first layer weights)
        g = ReLU  (nonlinearity)
        a ∈ R^k  (output weights)
        b = scalar bias

    This model can learn nonlinear functions of the input-representation combination,
    unlike the scalar head which only applies 1D nonlinearity.

    Architecture:
        1. Compute rep = γ * S @ X^T @ y  (same as linear model)
        2. Combine with test input: h^L = x* ⊙ rep / (LP)  (element-wise product)
        3. Apply 2-layer NN (pure TA's note, no skip):
           f = a^T @ ReLU(W_0 @ h^L + b_0) + b_1
    """

    def __init__(self, D: int, L: int, hidden_dim: int = 64, init_gamma: float = 1.0):
        super().__init__()
        self.D = D
        self.L = L
        self.hidden_dim = hidden_dim

        # Attention parameter - initialize to 1.0 for better gradient flow
        self.gamma = nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32))

        # 2-layer NN parameters
        # W_0: R^{k×D}, maps h^L to hidden layer
        self.W0 = nn.Parameter(torch.randn(hidden_dim, D) * 0.01)
        self.b0 = nn.Parameter(torch.zeros(hidden_dim))

        # a: R^k, output weights
        self.a = nn.Parameter(torch.randn(hidden_dim) * 0.01)

        # Output bias
        self.b1 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, X: torch.Tensor, y: torch.Tensor, X_star: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for 2-layer NN model.

        Implements TA's note exactly: f = a^T @ ReLU(W_0 @ h^L) + b

        Args:
            X: Training inputs (P, D)
            y: Training targets (P,)
            X_star: Test inputs (K, D)

        Returns:
            Predictions (K,)
        """
        P, D = X.shape
        L = self.L
        y = y.squeeze()

        # Same representation computation as linear model
        Sigma_hat = (X.T @ X) / P  # (D, D)
        I = torch.eye(D, device=X.device, dtype=X.dtype)
        M = I - (self.gamma / L) * Sigma_hat

        # Compute geometric series: S = Σ_{ℓ=0}^{L-1} M^ℓ
        S = torch.zeros_like(I)
        M_power = I.clone()
        for ell in range(L):
            S = S + M_power
            if ell < L - 1:
                M_power = M_power @ M

        # Representation: rep = γ * S @ X^T @ y  (D-dimensional)
        rep = self.gamma * (S @ (X.T @ y))  # (D,)

        # Combine with test inputs: h^L = x* ⊙ rep / (LP)
        h_L = (X_star * rep.unsqueeze(0)) / (L * P)  # (K, D)

        # Apply 2-layer NN (pure TA's note, no skip connection)
        # f = a^T @ ReLU(W_0 @ h^L + b_0) + b_1
        hidden = F.relu(h_L @ self.W0.T + self.b0)  # (K, hidden_dim)
        nn_output = hidden @ self.a  # (K,)

        predictions = nn_output + self.b1

        return predictions


class TwoLayerNNModelV2(nn.Module):
    """
    Alternative 2-Layer NN Model that transforms rep before combining with x*.

    Architecture:
        1. Compute rep = γ * S @ X^T @ y  (D-dimensional)
        2. Transform rep: transformed = W_1 @ ReLU(W_0 @ rep + b_0)  (D-dimensional)
        3. Combine with x* with skip: f(x*) = (x* · transformed) / (LP) + skip * linear_pred + b_1

    This model learns a nonlinear transformation of the representation,
    then combines it linearly with x* (similar to linear model's structure).
    """

    def __init__(self, D: int, L: int, hidden_dim: int = 64, init_gamma: float = 1.0):
        super().__init__()
        self.D = D
        self.L = L
        self.hidden_dim = hidden_dim

        # Attention parameter - initialize to 1.0
        self.gamma = nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32))

        # 2-layer NN that transforms rep (initialize small for residual-like behavior)
        self.W0 = nn.Parameter(torch.randn(hidden_dim, D) * 0.01)
        self.b0 = nn.Parameter(torch.zeros(hidden_dim))
        self.W1 = nn.Parameter(torch.randn(D, hidden_dim) * 0.01)

        # Skip connection (start at 1 to recover linear)
        self.skip_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.b1 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, X: torch.Tensor, y: torch.Tensor, X_star: torch.Tensor) -> torch.Tensor:
        P, D = X.shape
        L = self.L
        y = y.squeeze()

        # Compute representation (same as linear)
        Sigma_hat = (X.T @ X) / P
        I = torch.eye(D, device=X.device, dtype=X.dtype)
        M = I - (self.gamma / L) * Sigma_hat

        S = torch.zeros_like(I)
        M_power = I.clone()
        for ell in range(L):
            S = S + M_power
            if ell < L - 1:
                M_power = M_power @ M

        rep = self.gamma * (S @ (X.T @ y))  # (D,)

        # Linear prediction (for skip connection)
        linear_pred = (X_star @ rep) / (L * P)  # (K,)

        # Transform rep through 2-layer NN (nonlinear correction)
        hidden = F.relu(self.W0 @ rep + self.b0)  # (hidden_dim,)
        transformed_rep = self.W1 @ hidden  # (D,)

        # Combine with x* and add skip connection
        nn_pred = (X_star @ transformed_rep) / (L * P)  # (K,)
        predictions = nn_pred + self.skip_weight * linear_pred + self.b1

        return predictions


def generate_nonlinear_iso_data(
    D: int,
    P: int,
    K: int,
    sigma: float = 0.0,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate NON-LINEAR isotropic data.

    Data generation:
        β ~ N(0, I)
        x ~ N(0, I)
        z = β·x / √D
        y = ReLU(z) + σε

    Note: E[ReLU(z)] > 0 when z ~ N(0,1), so targets have positive mean.
    Models include bias to handle this.

    Args:
        D: Dimension
        P: Number of training points
        K: Number of test points
        sigma: Label noise standard deviation
        device: Device

    Returns:
        X, y, X_star, y_star, beta
    """
    # Sample task weights: β ~ N(0, I)
    beta = torch.randn(D, device=device)

    # Sample training inputs: x ~ N(0, I)
    X = torch.randn(P, D, device=device)

    # Sample test inputs
    X_star = torch.randn(K, D, device=device)

    # NON-LINEAR labels: y = ReLU(β·x / √D) + σε
    z_train = (X @ beta) / np.sqrt(D)
    z_test = (X_star @ beta) / np.sqrt(D)

    y = F.relu(z_train)
    y_star = F.relu(z_test)

    # Add noise
    if sigma > 0:
        y = y + sigma * torch.randn(P, device=device)
        y_star = y_star + sigma * torch.randn(K, device=device)

    return X, y, X_star, y_star, beta


def train_model_on_nonlinear_data(
    model_type: str,
    D: int,
    L: int,
    alpha: float = 1.0,
    n_steps: int = 6000,
    lr: float = 0.01,
    sigma: float = 0.0,
    batch_size: int = 8,
    hidden_dim: int = 64,
    eval_every: int = 20,
    n_eval_contexts: int = 100,
    device: str = 'cpu',
    seed: int = 42
) -> Tuple[List[float], List[float], nn.Module]:
    """
    Train model on NON-LINEAR data.

    Args:
        model_type: 'linear', 'two_layer_nn', or 'two_layer_nn_v2'
        D: Dimension
        L: Depth
        alpha: Context length ratio (P/D)
        n_steps: Number of training steps
        lr: Learning rate
        sigma: Label noise
        batch_size: Number of contexts per optimizer step
        hidden_dim: Hidden dimension for 2-layer NN
        eval_every: Evaluate every N steps
        n_eval_contexts: Number of contexts for evaluation
        device: Device
        seed: Random seed

    Returns:
        steps, losses, trained_model
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    P = max(1, int(alpha * D))
    K = max(1, int(alpha * D))

    # Initialize model
    if model_type == 'linear':
        model = LinearModelWithBias(D=D, L=L, init_gamma=0.0).to(device)
    elif model_type == 'two_layer_nn':
        model = TwoLayerNNModel(D=D, L=L, hidden_dim=hidden_dim, init_gamma=0.0).to(device)
    elif model_type == 'two_layer_nn_v2':
        model = TwoLayerNNModelV2(D=D, L=L, hidden_dim=hidden_dim, init_gamma=0.0).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    steps = []
    losses = []

    for step in range(n_steps):
        optimizer.zero_grad()

        # Average loss over batch_size contexts
        batch_loss = 0.0
        for _ in range(batch_size):
            X, y, X_star, y_star, _ = generate_nonlinear_iso_data(D, P, K, sigma, device)
            loss = compute_icl_loss(model, X, y, X_star, y_star)
            batch_loss = batch_loss + loss

        batch_loss = batch_loss / batch_size
        batch_loss.backward()
        optimizer.step()

        # Evaluate
        if step % eval_every == 0:
            model.eval()
            eval_loss = 0.0

            with torch.no_grad():
                for _ in range(n_eval_contexts):
                    X, y, X_star, y_star, _ = generate_nonlinear_iso_data(D, P, K, sigma, device)
                    loss_val = compute_icl_loss(model, X, y, X_star, y_star)
                    eval_loss += loss_val.item()

            eval_loss /= n_eval_contexts
            steps.append(step)
            losses.append(eval_loss)
            model.train()

    return steps, losses, model


def run_comparison(
    D: int = 64,
    n_steps: int = 6000,
    lr: float = 0.01,
    sigma: float = 0.0,
    batch_size: int = 8,
    hidden_dim: int = 64,
    device: str = 'cpu',
    save_path: str = None
):
    """
    Compare Linear vs 2-Layer NN on Non-Linear Data.

    Based on TA's suggestion:
    - Model 1 (Linear): f̂ = w^T · h^L
    - Model 2 (2-Layer NN): f̂ = a^T · g(W_0 · h^L)
    """
    print("=" * 70)
    print("Task 5 (v2): Linear vs 2-Layer NN on Non-Linear Data")
    print("=" * 70)
    print(f"\nData: y = ReLU(β·x / √D) + σε  (σ = {sigma})")
    print(f"Model 2: f̂ = a^T · ReLU(W_0 · h^L)  (proper 2-layer NN)")
    print(f"Hidden dim: {hidden_dim}, Batch size: {batch_size}")
    print(f"D = {D}, n_steps = {n_steps}, lr = {lr}")
    print("=" * 70)

    # Sweep over alpha values
    alphas = [1, 2, 4]
    depths = [1, 2, 4, 8, 16]
    colors = ['#1f77b4', '#9467bd', '#d62728', '#ff7f0e', '#2ca02c']

    all_results = {}

    for alpha in alphas:
        print(f"\n{'='*60}")
        print(f"Training with α = {alpha} (P = {alpha}D = {alpha * D})")
        print(f"{'='*60}")

        linear_results = {}
        nn_results = {}

        # Train Linear models
        print(f"\n  --- Linear Models (α={alpha}) ---")
        for L in depths:
            print(f"    L={L}...", end=" ", flush=True)
            steps, losses, _ = train_model_on_nonlinear_data(
                model_type='linear',
                D=D, L=L, alpha=alpha, n_steps=n_steps,
                lr=lr, sigma=sigma, batch_size=batch_size,
                eval_every=20, n_eval_contexts=100,
                device=device, seed=42
            )
            linear_results[L] = (steps, losses)
            print(f"loss={losses[-1]:.4f}")

        # Train 2-Layer NN models
        print(f"\n  --- 2-Layer NN Models (α={alpha}) ---")
        for L in depths:
            print(f"    L={L}...", end=" ", flush=True)
            steps, losses, _ = train_model_on_nonlinear_data(
                model_type='two_layer_nn',
                D=D, L=L, alpha=alpha, n_steps=n_steps,
                lr=lr, sigma=sigma, batch_size=batch_size,
                hidden_dim=hidden_dim,
                eval_every=20, n_eval_contexts=100,
                device=device, seed=42
            )
            nn_results[L] = (steps, losses)
            print(f"loss={losses[-1]:.4f}")

        all_results[alpha] = {'linear': linear_results, 'two_layer_nn': nn_results}

    # ========== Create Comparison Plot ==========
    print("\n" + "=" * 50)
    print("Creating comparison plot")
    print("=" * 50)

    fig, axes = plt.subplots(len(alphas), 2, figsize=(14, 4 * len(alphas)))

    for row, alpha in enumerate(alphas):
        linear_results = all_results[alpha]['linear']
        nn_results = all_results[alpha]['two_layer_nn']

        # Left: Linear model
        ax = axes[row, 0]
        for i, L in enumerate(depths):
            steps, losses = linear_results[L]
            ax.plot(steps, losses, color=colors[i], label=f'L={L}', linewidth=1.5)
        ax.set_xlabel('Training Step', fontsize=10)
        ax.set_ylabel('Loss (MSE)', fontsize=10)
        ax.set_title(f'Linear Model (α={alpha}, P={alpha*D})', fontsize=11)
        ax.legend(fontsize=9, loc='upper right')
        ax.set_ylim(0, 0.6)
        ax.grid(True, alpha=0.3)

        # Right: 2-Layer NN model
        ax = axes[row, 1]
        for i, L in enumerate(depths):
            steps, losses = nn_results[L]
            ax.plot(steps, losses, color=colors[i], label=f'L={L}', linewidth=1.5)
        ax.set_xlabel('Training Step', fontsize=10)
        ax.set_ylabel('Loss (MSE)', fontsize=10)
        ax.set_title(f'2-Layer NN (α={alpha}, P={alpha*D})', fontsize=11)
        ax.legend(fontsize=9, loc='upper right')
        ax.set_ylim(0, 0.6)
        ax.grid(True, alpha=0.3)

    plt.suptitle(r'Linear vs 2-Layer NN on Nonlinear Data ($y = \mathrm{ReLU}(\beta \cdot x / \sqrt{D})$)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved to {save_path}")
    else:
        plt.show()

    plt.close()

    # ========== Print Summary ==========
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for alpha in alphas:
        linear_results = all_results[alpha]['linear']
        nn_results = all_results[alpha]['two_layer_nn']

        print(f"\nα = {alpha} (P = {alpha * D}):")
        print("-" * 60)
        print(f"{'Depth L':<10} {'Linear':<12} {'2-Layer NN':<12} {'Improvement':<15}")
        print("-" * 60)
        for L in depths:
            linear_loss = linear_results[L][1][-1]
            nn_loss = nn_results[L][1][-1]
            improvement = (linear_loss - nn_loss) / linear_loss * 100
            print(f"{L:<10} {linear_loss:<12.4f} {nn_loss:<12.4f} {improvement:>+.1f}%")

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("- 2-Layer NN: f̂ = a^T · ReLU(W_0 · h^L)")
    print("- h^L = x* ⊙ rep / (LP) combines test input with context")
    print("- Positive improvement = 2-Layer NN outperforms Linear")
    print("- Unlike ScalarHead (1D nonlinearity), this uses D-dimensional nonlinearity")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    # Check for GPU
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs('results', exist_ok=True)

    # Run comparison
    all_results = run_comparison(
        D=64,
        n_steps=6000,
        lr=0.01,
        sigma=0.0,
        batch_size=8,
        hidden_dim=64,
        device=device,
        save_path='results/task5_linear_vs_2layer_nn.png'
    )

    print("\nTask 5 (v2) complete!")
    print("Results saved to: results/task5_linear_vs_2layer_nn2.png")
