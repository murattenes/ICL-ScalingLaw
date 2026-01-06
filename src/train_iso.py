"""
Training script for ISO setting experiments (Task 2).
Replicates Figure 1 from the paper.

Figure 1 shows:
(a) L=16, varying α (P/D ratio) from 0.2 to 8.0
(b) α=1, varying L (depth) from 1 to 16
(c) Final loss vs α for different depths, compared with theory
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict, Tuple
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import ReducedGammaModel, generate_iso_data, compute_icl_loss


def train_reduced_gamma_model(
    D: int,
    L: int,
    alpha: float,
    n_steps: int = 6000,
    lr: float = 0.01,
    batch_size_ratio: float = 1.0,  # B/D = τ
    eval_size_ratio: float = 1.0,   # K/D = κ
    sigma: float = 0.0,
    eval_every: int = 10,
    n_eval_contexts: int = 50,
    device: str = 'cpu',
    seed: int = 42
) -> Tuple[List[float], List[float]]:
    """
    Train the reduced gamma model using SGD.

    Args:
        D: Input dimension
        L: Depth
        alpha: Context length ratio (P/D)
        n_steps: Number of training steps
        lr: Learning rate
        batch_size_ratio: B/D ratio (τ)
        eval_size_ratio: K/D ratio (κ)
        sigma: Label noise
        eval_every: Evaluate every N steps
        n_eval_contexts: Number of contexts for evaluation
        device: Device to use
        seed: Random seed

    Returns:
        steps: List of step numbers
        losses: List of losses at each evaluation
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    P = int(alpha * D)  # Context length
    B = int(batch_size_ratio * D)  # Batch size (contexts per step)
    K = max(1, int(eval_size_ratio * D))  # Eval points per context

    # Initialize model
    model = ReducedGammaModel(D=D, L=L, init_gamma=0.0).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    steps = []
    losses = []

    for step in range(n_steps):
        # Training step: sample B contexts and update
        total_loss = 0.0

        for _ in range(B):
            # Generate a context
            X, y, X_star, y_star, _ = generate_iso_data(D, P, K, sigma, device)

            # Compute loss and accumulate gradients
            optimizer.zero_grad()
            loss = compute_icl_loss(model, X, y, X_star, y_star)
            loss.backward()
            total_loss += loss.item()

        # Average gradients and update
        for param in model.parameters():
            if param.grad is not None:
                param.grad /= B

        optimizer.step()

        # Evaluate
        if step % eval_every == 0:
            model.eval()
            eval_loss = 0.0

            with torch.no_grad():
                for _ in range(n_eval_contexts):
                    X, y, X_star, y_star, _ = generate_iso_data(D, P, K, sigma, device)
                    loss = compute_icl_loss(model, X, y, X_star, y_star)
                    eval_loss += loss.item()

            eval_loss /= n_eval_contexts
            steps.append(step)
            losses.append(eval_loss)
            model.train()

    return steps, losses


def train_reduced_gamma_fast(
    D: int,
    L: int,
    alpha: float,
    n_steps: int = 6000,
    lr: float = 0.01,
    sigma: float = 0.0,
    eval_every: int = 10,
    n_eval_contexts: int = 100,
    device: str = 'cpu',
    seed: int = 42
) -> Tuple[List[float], List[float], List[float]]:
    """
    Faster training using batched operations and gradient flow approximation.

    This version uses a single context per step but averages over many eval contexts.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    P = max(1, int(alpha * D))  # Context length
    K = max(1, int(alpha * D))  # Eval points

    # Initialize model
    model = ReducedGammaModel(D=D, L=L, init_gamma=0.0).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    steps = []
    losses = []
    gammas = []

    for step in range(n_steps):
        # Training step
        X, y, X_star, y_star, _ = generate_iso_data(D, P, K, sigma, device)

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
                    X, y, X_star, y_star, _ = generate_iso_data(D, P, K, sigma, device)
                    loss_val = compute_icl_loss(model, X, y, X_star, y_star)
                    eval_loss += loss_val.item()

            eval_loss /= n_eval_contexts
            steps.append(step)
            losses.append(eval_loss)
            gammas.append(model.gamma.item())
            model.train()

    return steps, losses, gammas


def compute_theory_loss_iso(gamma: float, L: int, alpha: float) -> float:
    """
    Compute theoretical loss for ISO setting using Marchenko-Pastur.

    L(γ) = E_λ[(1 - γλ/L)^{2L}]

    where λ follows Marchenko-Pastur distribution.
    """
    n_samples = 5000

    if alpha >= 1:
        # MP distribution bounds
        lambda_minus = (1 - 1/np.sqrt(alpha))**2
        lambda_plus = (1 + 1/np.sqrt(alpha))**2

        # Sample eigenvalues
        lambdas = np.linspace(lambda_minus + 1e-8, lambda_plus - 1e-8, n_samples)

        # MP density
        density = (alpha / (2 * np.pi)) * np.sqrt(
            np.maximum(0, (lambda_plus - lambdas) * (lambdas - lambda_minus))
        ) / np.maximum(lambdas, 1e-10)

        # Normalize
        dlambda = lambdas[1] - lambdas[0]
        density = density / (np.sum(density) * dlambda)

        # Compute loss
        residuals = (1 - gamma * lambdas / L) ** (2 * L)
        loss = np.sum(density * residuals) * dlambda

    else:
        # α < 1: point mass at 0
        lambda_minus = (1 - 1/np.sqrt(alpha))**2
        lambda_plus = (1 + 1/np.sqrt(alpha))**2
        mass_at_zero = 1 - alpha

        lambdas = np.linspace(lambda_minus + 1e-8, lambda_plus - 1e-8, n_samples)
        density = (alpha / (2 * np.pi)) * np.sqrt(
            np.maximum(0, (lambda_plus - lambdas) * (lambdas - lambda_minus))
        ) / np.maximum(lambdas, 1e-10)

        dlambda = lambdas[1] - lambdas[0]
        density = density / (np.sum(density) * dlambda)

        residuals = (1 - gamma * lambdas / L) ** (2 * L)
        continuous_loss = np.sum(density * residuals) * dlambda

        loss = mass_at_zero * 1.0 + alpha * continuous_loss

    return loss


def find_optimal_gamma_theory(L: int, alpha: float) -> Tuple[float, float]:
    """
    Find optimal gamma and corresponding loss for theory curve.
    """
    gammas = np.linspace(0.01, 2*L, 500)
    losses = [compute_theory_loss_iso(g, L, alpha) for g in gammas]
    min_idx = np.argmin(losses)
    return gammas[min_idx], losses[min_idx]


def run_figure_1a(D: int = 32, L: int = 16, n_steps: int = 6000,
                  device: str = 'cpu', save_path: str = None):
    """
    Replicate Figure 1(a): L=16, varying α.

    Training dynamics for varying context length ratios α.
    """
    print(f"Running Figure 1a: L={L}, D={D}")

    alphas = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(alphas)))

    results = {}

    for i, alpha in enumerate(alphas):
        print(f"  Training with α={alpha}...")
        steps, losses, _ = train_reduced_gamma_fast(
            D=D, L=L, alpha=alpha, n_steps=n_steps,
            lr=0.01, eval_every=20, n_eval_contexts=100,
            device=device, seed=42
        )
        results[alpha] = (steps, losses)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, alpha in enumerate(alphas):
        steps, losses = results[alpha]
        ax.plot(steps, losses, color=colors[i], label=f'α = {alpha}', linewidth=1.5)

    ax.set_xlabel('t', fontsize=12)
    ax.set_ylabel(r'$\mathcal{L}(t, \alpha)$', fontsize=12)
    ax.set_title(f'(a) L = {L}, Linear Transformer', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, n_steps)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved to {save_path}")
    else:
        plt.show()

    plt.close()

    return results


def run_figure_1b(D: int = 32, alpha: float = 1.0, n_steps: int = 6000,
                  device: str = 'cpu', save_path: str = None):
    """
    Replicate Figure 1(b): α=1, varying L.

    Training dynamics for varying depths.
    """
    print(f"Running Figure 1b: α={alpha}, D={D}")

    depths = [1, 2, 4, 8, 16]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(depths)))

    results = {}

    for i, L in enumerate(depths):
        print(f"  Training with L={L}...")
        steps, losses, _ = train_reduced_gamma_fast(
            D=D, L=L, alpha=alpha, n_steps=n_steps,
            lr=0.01, eval_every=20, n_eval_contexts=100,
            device=device, seed=42
        )
        results[L] = (steps, losses)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, L in enumerate(depths):
        steps, losses = results[L]
        ax.plot(steps, losses, color=colors[i], label=f'L = {L}', linewidth=1.5)

    ax.set_xlabel('t', fontsize=12)
    ax.set_ylabel(r'$\mathcal{L}(t, L)$', fontsize=12)
    ax.set_title(f'(b) α = {alpha}, Linear Transformer', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, n_steps)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved to {save_path}")
    else:
        plt.show()

    plt.close()

    return results


def run_figure_1c(D: int = 32, n_steps: int = 8000, device: str = 'cpu',
                  save_path: str = None):
    """
    Replicate Figure 1(c): Final loss vs α for different depths.

    Compares experimental results with theoretical predictions.
    """
    print(f"Running Figure 1c: D={D}")

    depths = [1, 2, 4, 8, 16]
    alphas = np.logspace(-1, 1, 15)  # 0.1 to 10

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(depths)))
    markers = ['o', 's', 'D', '^', 'v']

    experimental_results = {}
    theory_results = {}

    for i, L in enumerate(depths):
        print(f"  Computing for L={L}...")
        exp_losses = []
        theory_losses = []

        for alpha in tqdm(alphas, desc=f"    L={L}"):
            # Run experiment
            _, losses, _ = train_reduced_gamma_fast(
                D=D, L=L, alpha=alpha, n_steps=n_steps,
                lr=0.01, eval_every=50, n_eval_contexts=100,
                device=device, seed=42
            )
            # Take final loss (average of last few)
            final_loss = np.mean(losses[-5:])
            exp_losses.append(final_loss)

            # Compute theory
            _, theory_loss = find_optimal_gamma_theory(L, alpha)
            theory_losses.append(theory_loss)

        experimental_results[L] = exp_losses
        theory_results[L] = theory_losses

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot theory curves (solid lines)
    for i, L in enumerate(depths):
        ax.plot(alphas, theory_results[L], color=colors[i], linewidth=2,
                label=f'L={L}' if i == 0 else None)

    # Plot experimental points
    for i, L in enumerate(depths):
        ax.scatter(alphas, experimental_results[L], color=colors[i],
                   marker=markers[i], s=40, edgecolors='black', linewidths=0.5,
                   label=f'L={L}')

    # Add theory line to legend
    ax.plot([], [], 'k-', linewidth=2, label='Theory')

    ax.set_xscale('log')
    ax.set_xlabel(r'$\alpha$', fontsize=12)
    ax.set_ylabel(r'$\mathcal{L}(\alpha)$', fontsize=12)
    ax.set_title('(c) Final Loss of The Transformer', fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved to {save_path}")
    else:
        plt.show()

    plt.close()

    return experimental_results, theory_results


def run_all_figure_1(D: int = 32, device: str = 'cpu', output_dir: str = 'results'):
    """
    Run all parts of Figure 1 and create combined figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Replicating Figure 1: Deep Linear Attention on ISO Setting")
    print("=" * 60)

    # Run each part
    results_1a = run_figure_1a(D=D, L=16, n_steps=6000, device=device,
                               save_path=os.path.join(output_dir, 'figure_1a.png'))

    results_1b = run_figure_1b(D=D, alpha=1.0, n_steps=6000, device=device,
                               save_path=os.path.join(output_dir, 'figure_1b.png'))

    exp_results, theory_results = run_figure_1c(D=D, n_steps=8000, device=device,
                                                 save_path=os.path.join(output_dir, 'figure_1c.png'))

    # Create combined figure like in the paper
    print("\nCreating combined Figure 1...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) L=16, varying α
    alphas = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0]
    colors_a = plt.cm.viridis(np.linspace(0.2, 0.9, len(alphas)))
    for i, alpha in enumerate(alphas):
        steps, losses = results_1a[alpha]
        axes[0].plot(steps, losses, color=colors_a[i], label=f'α = {alpha}', linewidth=1.2)
    axes[0].set_xlabel('t', fontsize=11)
    axes[0].set_ylabel(r'$\mathcal{L}(t, \alpha)$', fontsize=11)
    axes[0].set_title('(a) L = 16, Linear Transformer', fontsize=11)
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3)

    # (b) α=1, varying L
    depths = [1, 2, 4, 8, 16]
    colors_b = plt.cm.plasma(np.linspace(0.1, 0.9, len(depths)))
    for i, L in enumerate(depths):
        steps, losses = results_1b[L]
        axes[1].plot(steps, losses, color=colors_b[i], label=f'L = {L}', linewidth=1.2)
    axes[1].set_xlabel('t', fontsize=11)
    axes[1].set_ylabel(r'$\mathcal{L}(t, L)$', fontsize=11)
    axes[1].set_title(r'(b) $\alpha$ = 1, Linear Transformer', fontsize=11)
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)

    # (c) Final loss vs α
    alphas_c = np.logspace(-1, 1, 15)
    markers = ['o', 's', 'D', '^', 'v']
    for i, L in enumerate(depths):
        axes[2].plot(alphas_c, theory_results[L], color=colors_b[i], linewidth=1.5)
        axes[2].scatter(alphas_c, exp_results[L], color=colors_b[i],
                        marker=markers[i], s=30, edgecolors='black', linewidths=0.5,
                        label=f'L={L}')
    axes[2].plot([], [], 'k-', linewidth=2, label='Theory')
    axes[2].set_xscale('log')
    axes[2].set_xlabel(r'$\alpha$', fontsize=11)
    axes[2].set_ylabel(r'$\mathcal{L}(\alpha)$', fontsize=11)
    axes[2].set_title('(c) Final Loss of The Transformer', fontsize=11)
    axes[2].legend(fontsize=8, ncol=2)
    axes[2].set_ylim(0, 1.0)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    combined_path = os.path.join(output_dir, 'figure_1_combined.png')
    plt.savefig(combined_path, dpi=200, bbox_inches='tight')
    print(f"Saved combined figure to {combined_path}")
    plt.close()

    print("\nFigure 1 replication complete!")


if __name__ == "__main__":
    # Check for GPU
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Run all experiments
    run_all_figure_1(D=32, device=device, output_dir='results')
