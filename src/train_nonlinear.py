"""
Training script for Task 5: Non-Linear Extension.

Compares Linear vs Scalar Head on Non-Linear Isotropic Data.

Key improvements implemented:
- Option A: 1D "two-sided ReLU" head on scalar u (cleanest for single-index data)
  Architecture: ŷ = u + c1*ReLU(u) + c2*ReLU(-u) + b
  where u = linear_pred is the scalar estimate of z = β·x/√D

- Option B: Sweep over α ∈ {1, 2, 4} (more context → better z estimate)

- Option C: Batch multiple contexts per optimizer step (reduces gradient variance)

Data:
- y = ReLU(β·x / √D) + σε (single-index nonlinear target)

Why Scalar Head works:
- Target depends on ONE scalar z, so learn 1D nonlinearity on scalar estimate u
- c1=c2=0 recovers linear exactly (strict superset)
- ReLU(u) = u - ReLU(-u), so head can represent ReLU efficiently
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


class ScalarHeadModel(nn.Module):
    """
    1D "two-sided ReLU" head on the scalar linear prediction.

    Architecture: ŷ = u + c1*ReLU(u) + c2*ReLU(-u) + b
    where u = linear_pred = (1/LP) x*^T @ rep

    This is the CLEANEST extension for single-index data y = ReLU(z):
    - The target depends on ONE scalar z = β·x/√D
    - We first estimate z via u (linear prediction)
    - Then learn a 1D nonlinearity u → y

    Key insight: ReLU(u) = u - ReLU(-u), so this head can represent ReLU efficiently.

    STRICT SUPERSET: Initialize c1=c2=0 → recovers linear model exactly.

    Trainable parameters: γ, c1, c2, b
    """

    def __init__(self, D: int, L: int, init_gamma: float = 0.0):
        super().__init__()
        self.D = D
        self.L = L

        # Trainable parameters
        self.gamma = nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32))
        self.c1 = nn.Parameter(torch.tensor(0.0))  # Coefficient for ReLU(u)
        self.c2 = nn.Parameter(torch.tensor(0.0))  # Coefficient for ReLU(-u)
        self.b = nn.Parameter(torch.tensor(0.0))   # Output bias

    def forward(self, X: torch.Tensor, y: torch.Tensor, X_star: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: u + c1*ReLU(u) + c2*ReLU(-u) + b

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

        # Linear prediction (scalar estimate of z)
        u = (X_star @ rep) / (L * P)  # (K,)

        # 1D two-sided ReLU head
        # ŷ = u + c1*ReLU(u) + c2*ReLU(-u) + b
        predictions = u + self.c1 * F.relu(u) + self.c2 * F.relu(-u) + self.b

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
    batch_size: int = 8,
    eval_every: int = 20,
    n_eval_contexts: int = 100,
    device: str = 'cpu',
    seed: int = 42
) -> Tuple[List[float], List[float]]:
    """
    Train either 'linear' or 'scalar_head' model on NON-LINEAR data.

    Args:
        model_type: 'linear' or 'scalar_head'
        D: Dimension
        L: Depth
        alpha: Context length ratio (P/D)
        n_steps: Number of training steps
        lr: Learning rate
        sigma: Label noise
        batch_size: Number of contexts to average per optimizer step (Option C)
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
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:  # 'scalar_head'
        # 1D scalar head: ŷ = u + c1*ReLU(u) + c2*ReLU(-u) + b
        model = ScalarHeadModel(D=D, L=L, init_gamma=0.0).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

    steps = []
    losses = []

    for step in range(n_steps):
        optimizer.zero_grad()

        # Option C: Average loss over batch_size contexts (reduces gradient variance)
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

    return steps, losses


def run_task5_comparison(
    D: int = 64,
    n_steps: int = 6000,
    lr: float = 0.01,
    sigma: float = 0.0,
    batch_size: int = 8,
    device: str = 'cpu',
    save_path: str = None
):
    """
    Task 5: Compare Linear vs Scalar Head on Non-Linear Data.

    Key improvements:
    - Option A: 1D two-sided ReLU head (cleanest for single-index data)
    - Option B: Sweep over α ∈ {1, 2, 4} to see effect of more context
    - Option C: Batch multiple contexts per step (batch_size > 1)

    Architecture: ŷ = u + c1*ReLU(u) + c2*ReLU(-u) + b
    where u is the linear prediction (scalar estimate of z).
    """
    print("=" * 70)
    print("Task 5: Linear vs Scalar Head on Non-Linear Data")
    print("=" * 70)
    print(f"\nData: y = ReLU(β·x / √D) + σε  (σ = {sigma})")
    print(f"Scalar Head: ŷ = u + c1·ReLU(u) + c2·ReLU(-u) + b  (strict superset)")
    print(f"Batch size: {batch_size} contexts per optimizer step")
    print(f"D = {D}, n_steps = {n_steps}, lr = {lr}")
    print("=" * 70)

    # Option B: Sweep over alpha values
    alphas = [1, 2, 4]
    depths = [1, 2, 4, 8, 16]
    colors = ['#1f77b4', '#9467bd', '#d62728', '#ff7f0e', '#2ca02c']

    all_results = {}

    for alpha in alphas:
        print(f"\n{'='*60}")
        print(f"Training with α = {alpha} (P = {alpha}D = {alpha * D})")
        print(f"{'='*60}")

        linear_results = {}
        scalar_results = {}

        # Train Linear models
        print(f"\n  --- Linear Models (α={alpha}) ---")
        for L in depths:
            print(f"    L={L}...", end=" ", flush=True)
            steps, losses = train_model_on_nonlinear_data(
                model_type='linear',
                D=D, L=L, alpha=alpha, n_steps=n_steps,
                lr=lr, sigma=sigma, batch_size=batch_size,
                eval_every=20, n_eval_contexts=100,
                device=device, seed=42
            )
            linear_results[L] = (steps, losses)
            print(f"loss={losses[-1]:.4f}")

        # Train Scalar Head models
        print(f"\n  --- Scalar Head Models (α={alpha}) ---")
        for L in depths:
            print(f"    L={L}...", end=" ", flush=True)
            steps, losses = train_model_on_nonlinear_data(
                model_type='scalar_head',
                D=D, L=L, alpha=alpha, n_steps=n_steps,
                lr=lr, sigma=sigma, batch_size=batch_size,
                eval_every=20, n_eval_contexts=100,
                device=device, seed=42
            )
            scalar_results[L] = (steps, losses)
            print(f"loss={losses[-1]:.4f}")

        all_results[alpha] = {'linear': linear_results, 'scalar': scalar_results}

    # ========== Create Comparison Plot ==========
    print("\n" + "=" * 50)
    print("Creating comparison plot")
    print("=" * 50)

    fig, axes = plt.subplots(len(alphas), 2, figsize=(14, 4 * len(alphas)))

    for row, alpha in enumerate(alphas):
        linear_results = all_results[alpha]['linear']
        scalar_results = all_results[alpha]['scalar']

        # Left: Linear model
        ax = axes[row, 0]
        for i, L in enumerate(depths):
            steps, losses = linear_results[L]
            ax.plot(steps, losses, color=colors[i], label=f'L={L}', linewidth=1.5)
        ax.set_xlabel('Training Step', fontsize=10)
        ax.set_ylabel('Loss (MSE)', fontsize=10)
        ax.set_title(f'Linear (α={alpha}, P={alpha*D})', fontsize=11)
        ax.legend(fontsize=9, loc='upper right')
        ax.set_ylim(0, 0.6)
        ax.grid(True, alpha=0.3)

        # Right: Scalar Head model
        ax = axes[row, 1]
        for i, L in enumerate(depths):
            steps, losses = scalar_results[L]
            ax.plot(steps, losses, color=colors[i], label=f'L={L}', linewidth=1.5)
        ax.set_xlabel('Training Step', fontsize=10)
        ax.set_ylabel('Loss (MSE)', fontsize=10)
        ax.set_title(f'Scalar Head (α={alpha}, P={alpha*D})', fontsize=11)
        ax.legend(fontsize=9, loc='upper right')
        ax.set_ylim(0, 0.6)
        ax.grid(True, alpha=0.3)

    plt.suptitle(r'Task 5: Linear vs Scalar Head on Nonlinear Data ($y = \mathrm{ReLU}(\beta \cdot x / \sqrt{D})$)',
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
        scalar_results = all_results[alpha]['scalar']

        print(f"\nα = {alpha} (P = {alpha * D}):")
        print("-" * 55)
        print(f"{'Depth L':<10} {'Linear':<12} {'Scalar':<12} {'Improvement':<15}")
        print("-" * 55)
        for L in depths:
            linear_loss = linear_results[L][1][-1]
            scalar_loss = scalar_results[L][1][-1]
            improvement = (linear_loss - scalar_loss) / linear_loss * 100
            print(f"{L:<10} {linear_loss:<12.4f} {scalar_loss:<12.4f} {improvement:>+.1f}%")

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("- Scalar Head: ŷ = u + c1·ReLU(u) + c2·ReLU(-u) + b")
    print("- c1=c2=0 recovers linear (strict superset)")
    print("- Larger α → better z estimate → larger improvement from nonlinear head")
    print("- Positive improvement = Scalar Head outperforms Linear")
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

    # Run Task 5 comparison with all improvements:
    # - Option A: 1D scalar head (two-sided ReLU)
    # - Option B: Sweep α ∈ {1, 2, 4}
    # - Option C: Batch 8 contexts per step
    all_results = run_task5_comparison(
        D=64,
        n_steps=6000,  # Fewer steps since batch_size=8 (effectively 48k context updates)
        lr=0.01,
        sigma=0.0,
        batch_size=8,
        device=device,
        save_path='results/task5_linear_vs_scalar_head.png'
    )

    print("\nTask 5 complete!")
    print("Results saved to: results/task5_linear_vs_scalar_head.png")
