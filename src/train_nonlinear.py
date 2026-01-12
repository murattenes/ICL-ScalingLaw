"""
Training script for Task 5: Non-Linear Extension.

Compares Linear vs 2-Layer Neural Network on Non-Linear Isotropic Data.

Key idea from TA:
- Paper's linear model: f(x*) = (1/LP) x*^T @ rep
- 2-Layer NN extension: F(x*) = (1/LP) a^T @ ReLU(W @ (x* ⊙ rep))

Data:
- Linear: y = β·x / √D + σε (paper's ISO setting)
- Nonlinear: y = ReLU(β·x / √D) + σε (non-centered, both models have bias for fair comparison)

Experiment:
- Train both models on nonlinear data
- Compare across depths L = 1, 2, 4, 8, 16
- Hypothesis: 2-Layer should achieve lower loss on nonlinear data
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

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import ReducedGammaModel, compute_icl_loss


class ReducedGammaModelWithBias(ReducedGammaModel):
    """
    ReducedGammaModel with an added output bias for non-centered targets.

    Architecture: f(x*) = (1/LP) x*^T @ rep + bias

    The bias is NOT divided by (L*P) - it's an intercept in output space.
    """

    def __init__(self, D: int, L: int, init_gamma: float = 0.0, init_bias: float = 0.0):
        super().__init__(D=D, L=L, init_gamma=init_gamma)
        self.bias = nn.Parameter(torch.tensor(init_bias, dtype=torch.float32))

    def forward(self, X, y, X_star):
        return super().forward(X, y, X_star) + self.bias


class TwoLayerReducedGammaModel(nn.Module):
    """
    2-layer NN extension of ReducedGammaModel with SKIP CONNECTION.

    Architecture: F(x*) = linear_pred + mlp_correction + b2
    where:
        - linear_pred = (1/LP) x*^T @ rep  (original reduced-Γ output)
        - mlp_correction = a^T @ ReLU(W @ φ_normalized + b1)
        - φ_normalized = φ / ||rep|| where φ = x* ⊙ rep

    This is a STRICT SUPERSET of the linear model:
    - If a=0, b2=0, we recover the linear model exactly.

    Trainable parameters: γ, W, a, b1, b2
    - b1: hidden bias (helps learn thresholds)
    - b2: output bias (intercept in output space)
    """

    def __init__(self, D: int, L: int, hidden_dim: int = None, init_gamma: float = 0.001):
        super().__init__()
        self.D = D
        self.L = L
        # Use hidden_dim >= 2D for expressivity
        m = hidden_dim if hidden_dim else 2 * D

        # ALL TRAINABLE parameters
        self.gamma = nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32))
        self.W = nn.Parameter(torch.randn(m, D) * 0.01)  # Smaller init
        self.a = nn.Parameter(torch.zeros(m))  # Start at zero: recovers linear model
        self.b1 = nn.Parameter(torch.zeros(m))  # Hidden bias
        self.b2 = nn.Parameter(torch.tensor(0.0))  # Output bias (intercept)

    def forward(self, X: torch.Tensor, y: torch.Tensor, X_star: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with skip connection: linear_pred + mlp_correction.

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

        # Representation: rep = γ * S @ X^T @ y
        rep = self.gamma * (S @ (X.T @ y))  # (D,)

        # === LINEAR PREDICTION (skip connection) ===
        # This is exactly the original reduced-Γ model output
        linear_pred = (X_star @ rep) / (L * P)  # (K,)

        # === MLP CORRECTION ===
        # Normalize rep to stabilize gradients across depths
        rep_norm = torch.norm(rep) + 1e-8
        rep_normalized = rep / rep_norm

        # φ(x*) = x* ⊙ rep_normalized (element-wise product)
        phi = X_star * rep_normalized  # (K, D) broadcasting

        # Hidden layer with bias: ReLU(W @ φ + b1)
        hidden = F.relu(phi @ self.W.T + self.b1)  # (K, m)

        # MLP correction (no 1/LP division since we normalized)
        mlp_correction = hidden @ self.a  # (K,)

        # Final: skip + correction + bias
        predictions = linear_pred + mlp_correction + self.b2  # (K,)

        return predictions


def generate_nonlinear_iso_data(
    D: int,
    P: int,
    K: int,
    sigma: float = 0.0,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate NON-LINEAR isotropic data (non-centered).

    Data generation (extension of paper's ISO setting):
        β ~ N(0, I)
        x ~ N(0, I)
        z = β·x / √D
        y = ReLU(z) + σε

    Note: y has positive mean E[ReLU(z)] > 0. Both models (linear and 2-layer)
    include bias terms to handle this fairly.

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

    # Add noise (same mechanism as paper)
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
    eval_every: int = 20,
    n_eval_contexts: int = 100,
    device: str = 'cpu',
    seed: int = 42
) -> Tuple[List[float], List[float]]:
    """
    Train either 'linear' or '2layer' model on NON-LINEAR data.

    Args:
        model_type: 'linear' or '2layer'
        D: Dimension
        L: Depth
        alpha: Context length ratio (P/D)
        n_steps: Number of training steps
        lr: Learning rate
        sigma: Label noise
        eval_every: Evaluate every N steps
        n_eval_contexts: Number of contexts for evaluation
        device: Device
        seed: Random seed

    Returns:
        steps: List of step numbers
        losses: List of losses
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    P = max(1, int(alpha * D))  # Context length
    K = max(1, int(alpha * D))  # Test points

    # Initialize model (both have bias for fair comparison on non-centered data)
    if model_type == 'linear':
        model = ReducedGammaModelWithBias(D=D, L=L, init_gamma=0.0, init_bias=0.0).to(device)
        # Linear model uses SGD (simple convex problem)
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:  # '2layer'
        # 2-layer model: hidden_dim=2D by default, init gamma near good regime
        model = TwoLayerReducedGammaModel(D=D, L=L, init_gamma=0.5).to(device)
        # Use Adam with separate learning rates:
        # - Smaller lr for gamma (the representation parameter)
        # - Larger lr for MLP parameters (W, a, b1, b2)
        optimizer = optim.Adam([
            {'params': [model.gamma], 'lr': lr * 0.1},  # Smaller lr for gamma
            {'params': [model.W, model.a, model.b1, model.b2], 'lr': lr}  # Full lr for MLP
        ])

    steps = []
    losses = []

    for step in range(n_steps):
        # Generate NON-LINEAR data
        X, y, X_star, y_star, _ = generate_nonlinear_iso_data(D, P, K, sigma, device)

        # Training step
        optimizer.zero_grad()
        loss = compute_icl_loss(model, X, y, X_star, y_star)
        loss.backward()
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

    return steps, losses


def run_task5_comparison(
    D: int = 64,
    n_steps: int = 6000,
    lr: float = 0.01,
    sigma: float = 0.0,
    device: str = 'cpu',
    save_path: str = None
):
    """
    Task 5: Compare Linear vs 2-Layer NN on Non-Linear Data.

    Experiment:
    1. Train Linear model (ReducedGammaModel) on nonlinear data
    2. Train 2-Layer NN (TwoLayerReducedGammaModel) on same data
    3. Compare loss curves for depths L = 1, 2, 4, 8, 16
    4. Plot side-by-side comparison

    Hypothesis: 2-Layer NN should achieve lower loss on non-linear data.
    """
    print("=" * 70)
    print("Task 5: Linear vs 2-Layer NN on Non-Linear Data")
    print("=" * 70)
    print(f"\nData: y = ReLU(β·x / √D) + σε  (σ = {sigma})")
    print(f"2-Layer NN: skip connection + MLP correction (strict superset of linear)")
    print(f"Optimizer: Linear=SGD, 2-Layer=Adam (separate lr for γ vs MLP)")
    print(f"D = {D}, n_steps = {n_steps}, lr = {lr}")
    print("=" * 70)

    depths = [1, 2, 4, 8, 16]
    # Use distinct colors for better contrast
    colors = ['#1f77b4', '#9467bd', '#d62728', '#ff7f0e', '#2ca02c']

    linear_results = {}
    twolayer_results = {}

    # ========== Phase 1: Train Linear Models ==========
    print("\n" + "=" * 50)
    print("Phase 1: Training LINEAR models on nonlinear data")
    print("=" * 50)

    for i, L in enumerate(depths):
        print(f"\n  Training Linear L={L}...")
        steps, losses = train_model_on_nonlinear_data(
            model_type='linear',
            D=D, L=L, alpha=1.0, n_steps=n_steps,
            lr=lr, sigma=sigma, eval_every=20, n_eval_contexts=100,
            device=device, seed=42
        )
        linear_results[L] = (steps, losses)
        print(f"    Final loss: {losses[-1]:.4f}")

    # ========== Phase 2: Train 2-Layer Models ==========
    print("\n" + "=" * 50)
    print("Phase 2: Training 2-LAYER NN models on nonlinear data")
    print("=" * 50)

    for i, L in enumerate(depths):
        print(f"\n  Training 2-Layer L={L}...")
        steps, losses = train_model_on_nonlinear_data(
            model_type='2layer',
            D=D, L=L, alpha=1.0, n_steps=n_steps,
            lr=lr, sigma=sigma, eval_every=20, n_eval_contexts=100,
            device=device, seed=42
        )
        twolayer_results[L] = (steps, losses)
        print(f"    Final loss: {losses[-1]:.4f}")

    # ========== Phase 3: Create Comparison Plot ==========
    print("\n" + "=" * 50)
    print("Phase 3: Creating comparison plot")
    print("=" * 50)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Linear model
    ax = axes[0]
    for i, L in enumerate(depths):
        steps, losses = linear_results[L]
        ax.plot(steps, losses, color=colors[i], label=f'L = {L}', linewidth=1.5)
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Linear Model (γ + bias, SGD)\non Nonlinear Data', fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)

    # Right: 2-Layer model
    ax = axes[1]
    for i, L in enumerate(depths):
        steps, losses = twolayer_results[L]
        ax.plot(steps, losses, color=colors[i], label=f'L = {L}', linewidth=1.5)
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('2-Layer NN (skip + MLP, Adam)\non Nonlinear Data', fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)

    plt.suptitle(r'Task 5: Linear vs 2-Layer NN on Nonlinear Data ($y = \mathrm{ReLU}(\beta \cdot x / \sqrt{D})$)',
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

    print("\nFinal Loss Comparison (Linear vs 2-Layer):")
    print("-" * 50)
    print(f"{'Depth L':<10} {'Linear':<15} {'2-Layer':<15} {'Improvement':<15}")
    print("-" * 50)
    for L in depths:
        linear_loss = linear_results[L][1][-1]
        twolayer_loss = twolayer_results[L][1][-1]
        improvement = (linear_loss - twolayer_loss) / linear_loss * 100
        print(f"{L:<10} {linear_loss:<15.4f} {twolayer_loss:<15.4f} {improvement:>+.1f}%")

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("- 2-Layer NN is a STRICT SUPERSET of linear (a=0 recovers linear)")
    print("- 2-Layer should achieve <= loss of linear (by design)")
    print("- Skip connection ensures stable training while MLP learns corrections")
    print("- Positive improvement = 2-Layer outperforms linear on nonlinear data")
    print("=" * 70)

    return linear_results, twolayer_results


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

    # Run Task 5 comparison
    linear_results, twolayer_results = run_task5_comparison(
        D=64,
        n_steps=12000,
        lr=0.01,
        sigma=0.0,  # No noise for cleaner comparison
        device=device,
        save_path='results/task5_linear_vs_2layer.png'
    )

    print("\nTask 5 complete!")
    print("Results saved to: results/task5_linear_vs_2layer.png")
