"""
Training script for FS (Fixed Structured) setting experiments (Task 3).
Replicates Figure 3(c) from the paper: Brittleness to Distribution Shift.

Figure 3(c) shows:
- L_OOD (out-of-distribution loss) vs theta (rotation angle)
- For different depths L = 1, 2, 4, 8
- Deeper models are MORE brittle to distribution shift

Key concepts:
- Train on fixed covariance Sigma with power-law eigenvalues
- At convergence, model learns Gamma -> L * Sigma^{-1}
- Evaluate on rotated covariance Sigma' = exp(theta*S) * Sigma * exp(-theta*S)
- Higher L = more sensitivity to mismatch (brittleness)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, List, Dict
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class ReducedGammaModelFS(nn.Module):
    """
    Reduced Gamma Model for Fixed Structured (FS) covariance setting.

    Unlike ISO setting where Gamma = gamma * I (scalar),
    here Gamma is a full D x D matrix to learn the structure of Sigma.

    At convergence: Gamma -> L * Sigma^{-1}
    """

    def __init__(self, D: int, L: int = 1, init_scale: float = 0.01,
                 Sigma_init: torch.Tensor = None):
        """
        Args:
            D: Dimension
            L: Depth
            init_scale: Scale for random initialization (if Sigma_init is None)
            Sigma_init: If provided, initialize Gamma = L * Sigma^{-1} + noise
                        This is the theoretical optimal for FS setting.
        """
        super().__init__()
        self.D = D
        self.L = L

        if Sigma_init is not None:
            # Initialize near theoretical optimum: Gamma = L * Sigma^{-1}
            Sigma_inv = torch.linalg.inv(Sigma_init + 1e-6 * torch.eye(D))
            init_Gamma = L * Sigma_inv + torch.randn(D, D) * init_scale
            self.Gamma = nn.Parameter(init_Gamma.float())
        else:
            # Random initialization
            self.Gamma = nn.Parameter(torch.randn(D, D) * init_scale)

    def forward(self, X: torch.Tensor, y: torch.Tensor, X_star: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing Equation 4 with full Gamma matrix.

        f(x*) = (1/LP) x*^T Gamma Sum_{l=0}^{L-1} (I - L^{-1} Sigma_hat Gamma)^l X^T y

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

        # Compute empirical covariance: Sigma_hat = (1/P) X^T X
        Sigma_hat = (X.T @ X) / P  # (D, D)

        # Compute M = I - (1/L) * Sigma_hat @ Gamma
        I = torch.eye(D, device=X.device, dtype=X.dtype)
        M = I - (Sigma_hat @ self.Gamma) / L

        # Compute geometric series: S = Sum_{l=0}^{L-1} M^l
        S = torch.zeros_like(I)
        M_power = I.clone()
        for ell in range(L):
            S = S + M_power
            if ell < L - 1:
                M_power = M_power @ M

        # Compute X^T y
        Xy = X.T @ y  # (D,)

        # Compute Gamma @ S @ X^T y
        Gamma_S_Xy = self.Gamma @ (S @ Xy)  # (D,)

        # Compute predictions: f(x*) = (1/LP) x*^T Gamma S X^T y
        predictions = (X_star @ Gamma_S_Xy) / (L * P)  # (K,)

        return predictions


def create_powerlaw_covariance(D: int, nu: float = 1.0, device: str = 'cpu') -> torch.Tensor:
    """
    Create a power-law covariance matrix Sigma with eigenvalues lambda_k ~ k^{-nu}.

    Args:
        D: Dimension
        nu: Power-law exponent (default 1.0 as in paper)
        device: Device

    Returns:
        Sigma: (D, D) covariance matrix
    """
    # Eigenvalues: lambda_k = k^{-nu}
    eigenvalues = torch.tensor([(k + 1) ** (-nu) for k in range(D)],
                                dtype=torch.float32, device=device)
    # Sigma is diagonal in the canonical basis
    Sigma = torch.diag(eigenvalues)
    return Sigma


def create_powerlaw_task_covariance(D: int, beta: float = 1.0, nu: float = 1.0,
                                     device: str = 'cpu') -> torch.Tensor:
    """
    Create task covariance Omega with omega_k * lambda_k ~ k^{-nu*beta - 1}.

    From paper: omega_k ~ k^{-nu*beta - 1} / lambda_k = k^{-nu*beta - 1 + nu}

    Args:
        D: Dimension
        beta: Source exponent
        nu: Capacity exponent
        device: Device

    Returns:
        Omega: (D, D) task covariance matrix
    """
    # omega_k such that omega_k * lambda_k ~ k^{-nu*beta - 1}
    # lambda_k = k^{-nu}, so omega_k = k^{-nu*beta - 1 + nu}
    eigenvalues = torch.tensor([(k + 1) ** (-nu * beta - 1 + nu) for k in range(D)],
                                dtype=torch.float32, device=device)
    Omega = torch.diag(eigenvalues)
    return Omega


def generate_fs_data(D: int, P: int, K: int, Sigma: torch.Tensor, Omega: torch.Tensor,
                      device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor,
                                                     torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate data for the FS (Fixed Structured) setting.

    x ~ N(0, Sigma)
    beta ~ N(0, Omega)
    y = (1/sqrt(D)) beta . x

    Args:
        D: Dimension
        P: Number of training points
        K: Number of test points
        Sigma: Data covariance (D, D)
        Omega: Task covariance (D, D)
        device: Device

    Returns:
        X, y, X_star, y_star, beta
    """
    # Sample task vector beta ~ N(0, Omega)
    # Use Cholesky to sample: beta = Omega^{1/2} @ z where z ~ N(0, I)
    Omega_sqrt = torch.linalg.cholesky(Omega + 1e-6 * torch.eye(D, device=device))
    z = torch.randn(D, device=device)
    beta = Omega_sqrt @ z

    # Sample training inputs x ~ N(0, Sigma)
    Sigma_sqrt = torch.linalg.cholesky(Sigma + 1e-6 * torch.eye(D, device=device))
    Z_train = torch.randn(P, D, device=device)
    X = Z_train @ Sigma_sqrt.T  # (P, D)

    # Sample test inputs
    Z_test = torch.randn(K, D, device=device)
    X_star = Z_test @ Sigma_sqrt.T  # (K, D)

    # Compute targets: y = (1/sqrt(D)) beta . x
    y = (X @ beta) / np.sqrt(D)
    y_star = (X_star @ beta) / np.sqrt(D)

    return X, y, X_star, y_star, beta


def rotate_covariance(Sigma: torch.Tensor, theta: float, S: torch.Tensor) -> torch.Tensor:
    """
    Rotate covariance matrix: Sigma' = exp(theta*S) @ Sigma @ exp(-theta*S)

    where S is a skew-symmetric matrix.

    Args:
        Sigma: Original covariance (D, D)
        theta: Rotation angle
        S: Skew-symmetric matrix (D, D)

    Returns:
        Sigma': Rotated covariance (D, D)
    """
    # For MPS device, matrix_exp is not implemented, so move to CPU temporarily
    device = Sigma.device
    if device.type == 'mps':
        # Move tensors to CPU for matrix_exp computation
        S_cpu = S.cpu()
        Sigma_cpu = Sigma.cpu()
        # Compute rotation matrix R = exp(theta * S) on CPU
        R_cpu = torch.linalg.matrix_exp(theta * S_cpu)
        # Rotate: Sigma' = R @ Sigma @ R^T on CPU
        Sigma_prime_cpu = R_cpu @ Sigma_cpu @ R_cpu.T
        # Move back to original device
        Sigma_prime = Sigma_prime_cpu.to(device)
    else:
        # Compute rotation matrix R = exp(theta * S)
        R = torch.linalg.matrix_exp(theta * S)
        # Rotate: Sigma' = R @ Sigma @ R^T
        Sigma_prime = R @ Sigma @ R.T

    return Sigma_prime


def create_random_skew_symmetric(D: int, device: str = 'cpu', seed: int = 123) -> torch.Tensor:
    """
    Create a random skew-symmetric matrix S (S^T = -S).

    Args:
        D: Dimension
        device: Device
        seed: Random seed for reproducibility

    Returns:
        S: Skew-symmetric matrix (D, D)
    """
    torch.manual_seed(seed)
    A = torch.randn(D, D, device=device)
    S = (A - A.T) / 2  # Make it skew-symmetric
    # Normalize for stability
    S = S / torch.norm(S) * np.sqrt(D)
    return S


def compute_icl_loss(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                     X_star: torch.Tensor, y_star: torch.Tensor) -> torch.Tensor:
    """Compute ICL MSE loss."""
    predictions = model(X, y, X_star)
    loss = torch.mean((predictions - y_star) ** 2)
    return loss


def train_fs_model(
    D: int,
    L: int,
    Sigma: torch.Tensor,
    Omega: torch.Tensor,
    n_steps: int = 5000,
    lr: float = 0.001,
    alpha: float = 2.0,  # P/D ratio
    eval_every: int = 100,
    n_eval_contexts: int = 50,
    device: str = 'cpu',
    seed: int = 42,
    verbose: bool = True,
    init_at_optimal: bool = False
) -> Tuple[ReducedGammaModelFS, List[float], List[float]]:
    """
    Train ReducedGammaModelFS on fixed covariance data.

    Args:
        D: Dimension
        L: Depth
        Sigma: Data covariance
        Omega: Task covariance
        n_steps: Training steps
        lr: Learning rate
        alpha: Context length ratio P/D
        eval_every: Evaluation frequency
        n_eval_contexts: Contexts for evaluation
        device: Device
        seed: Random seed
        verbose: Print progress
        init_at_optimal: If True, initialize Gamma near L*Sigma^{-1}

    Returns:
        model: Trained model
        steps: List of step numbers
        losses: List of losses
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    P = max(1, int(alpha * D))
    K = max(1, int(alpha * D))

    Sigma = Sigma.to(device)
    Omega = Omega.to(device)

    # Initialize model
    if init_at_optimal:
        # Initialize near theoretical optimum for faster convergence
        model = ReducedGammaModelFS(D=D, L=L, init_scale=0.001,
                                     Sigma_init=Sigma.cpu()).to(device)
    else:
        model = ReducedGammaModelFS(D=D, L=L, init_scale=0.01).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    steps = []
    losses = []

    iterator = tqdm(range(n_steps), desc=f"L={L}") if verbose else range(n_steps)

    for step in iterator:
        # Generate training data
        X, y, X_star, y_star, _ = generate_fs_data(D, P, K, Sigma, Omega, device)

        # Training step
        optimizer.zero_grad()
        loss = compute_icl_loss(model, X, y, X_star, y_star)

        if torch.isnan(loss):
            continue

        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Evaluate
        if step % eval_every == 0:
            model.eval()
            eval_loss = 0.0

            with torch.no_grad():
                for _ in range(n_eval_contexts):
                    X, y, X_star, y_star, _ = generate_fs_data(D, P, K, Sigma, Omega, device)
                    loss_val = compute_icl_loss(model, X, y, X_star, y_star)
                    eval_loss += loss_val.item()

            eval_loss /= n_eval_contexts
            steps.append(step)
            losses.append(eval_loss)

            if verbose:
                iterator.set_postfix(loss=f"{eval_loss:.4f}")

            model.train()

    return model, steps, losses


def evaluate_ood_loss(
    model: ReducedGammaModelFS,
    Sigma_prime: torch.Tensor,
    Omega_prime: torch.Tensor,
    D: int,
    alpha: float = 2.0,
    n_contexts: int = 100,
    device: str = 'cpu'
) -> float:
    """
    Evaluate model on out-of-distribution data with rotated covariance.

    Args:
        model: Trained model
        Sigma_prime: Rotated data covariance
        Omega_prime: Rotated task covariance
        D: Dimension
        alpha: Context length ratio
        n_contexts: Number of contexts for evaluation
        device: Device

    Returns:
        Average OOD loss
    """
    model.eval()
    P = max(1, int(alpha * D))
    K = max(1, int(alpha * D))

    total_loss = 0.0

    with torch.no_grad():
        for _ in range(n_contexts):
            X, y, X_star, y_star, _ = generate_fs_data(
                D, P, K, Sigma_prime, Omega_prime, device
            )
            loss = compute_icl_loss(model, X, y, X_star, y_star)
            total_loss += loss.item()

    return total_loss / n_contexts


def run_figure_3c_experimental(
    D: int = 32,
    n_train_steps: int = 1000,
    lr: float = 0.0005,
    nu: float = 1.0,
    beta: float = 1.0,
    alpha: float = 4.0,
    device: str = 'cpu',
    save_path: str = None,
    init_at_optimal: bool = True
):
    """
    Experimentally reproduce Figure 3(c): Brittleness to Distribution Shift.

    This trains actual models and evaluates them on rotated covariances.

    Key insight from paper:
    - At convergence, Gamma = L * Sigma^{-1}
    - The paper's Figure 3c assumes perfect convergence
    - For experimental version, we initialize near optimal to ensure convergence

    NOTE ON init_at_optimal (from review):
    - When init_at_optimal=True (default), we initialize Γ≈LΣ⁻¹ and use Adam
    - This is primarily a CONVERGENCE CHECK to verify Result 5 (OOD loss formula)
    - It is NOT "training from zero" as described in Results 4/3(a,b)
    - For dynamics experiments (tracking gradient flow from scratch), set init_at_optimal=False
    - The paper's theoretical Figure 3(c) assumes perfect convergence anyway

    Args:
        D: Dimension
        n_train_steps: Training steps per model
        lr: Learning rate
        nu: Power-law exponent for covariance
        beta: Source exponent for task
        alpha: Context length ratio P/D (use large alpha for convergence)
        device: Device
        save_path: Path to save figure
        init_at_optimal: Initialize Gamma near L*Sigma^{-1} for faster convergence.
                         Use True for convergence checks (demonstrating Result 5).
                         Use False for training dynamics experiments.
    """
    print("=" * 60)
    print("Figure 3(c): Brittleness to Distribution Shift (Experimental)")
    print("=" * 60)

    # Create fixed covariance matrices
    Sigma = create_powerlaw_covariance(D, nu=nu, device=device)
    Omega = create_powerlaw_task_covariance(D, beta=beta, nu=nu, device=device)

    # Create random skew-symmetric matrix for rotation
    S = create_random_skew_symmetric(D, device=device, seed=123)

    # Depths to test
    depths = [1, 2, 4, 8]

    # Rotation angles theta
    thetas = np.linspace(0, 0.25, 30)

    # Train models for each depth
    trained_models = {}
    training_curves = {}

    print(f"\nPhase 1: Training models on fixed covariance (init_at_optimal={init_at_optimal})...")
    for L in depths:
        print(f"\n  Training L={L}...")
        model, steps, losses = train_fs_model(
            D=D, L=L, Sigma=Sigma, Omega=Omega,
            n_steps=n_train_steps, lr=lr, alpha=alpha,
            eval_every=50, n_eval_contexts=50,
            device=device, seed=42, verbose=True,
            init_at_optimal=init_at_optimal
        )
        trained_models[L] = model
        training_curves[L] = (steps, losses)
        print(f"    Final training loss: {losses[-1]:.6f}")

    # Evaluate OOD loss for each theta
    print("\nPhase 2: Evaluating OOD loss vs rotation angle...")
    ood_results = {L: [] for L in depths}

    for theta in tqdm(thetas, desc="Evaluating"):
        # Rotate covariance
        Sigma_prime = rotate_covariance(Sigma, theta, S)
        Omega_prime = rotate_covariance(Omega, theta, S)

        for L in depths:
            ood_loss = evaluate_ood_loss(
                trained_models[L], Sigma_prime, Omega_prime,
                D=D, alpha=alpha, n_contexts=100, device=device
            )
            ood_results[L].append(ood_loss)

    # Plot results
    print("\nCreating figure...")
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {1: '#1f1f1f', 2: '#8b008b', 4: '#b22222', 8: '#ff8c00'}

    for L in depths:
        ax.plot(thetas, ood_results[L], color=colors[L],
                label=f'L = {L}', linewidth=2)

    ax.set_xlabel(r'$\theta$', fontsize=14)
    ax.set_ylabel(r'$\mathcal{L}_{OOD}$', fontsize=14)
    ax.set_title('(c) Brittleness to Distribution Shift (Experimental)', fontsize=12)
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.25)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()

    # Also create training curves plot
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for L in depths:
        steps, losses = training_curves[L]
        ax2.plot(steps, losses, color=colors[L], label=f'L = {L}', linewidth=1.5)
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training Curves (FS Setting)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    if save_path:
        train_path = save_path.replace('.png', '_training.png')
        plt.savefig(train_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {train_path}")

    plt.close()

    return ood_results, training_curves


def compute_theoretical_ood_loss(
    Sigma: torch.Tensor,
    Omega: torch.Tensor,
    theta: float,
    S: torch.Tensor,
    L: int,
    device: str = 'cpu'
) -> float:
    """
    Compute theoretical OOD loss using Result 5 from the paper.

    L_OOD = tr Omega' [(I - Sigma^{-1} Sigma')^L]^T Sigma' (I - Sigma^{-1} Sigma')^L

    At convergence, Gamma = L * Sigma^{-1}, so the residual is (I - Sigma^{-1} Sigma')^L

    Args:
        Sigma: Training covariance
        Omega: Training task covariance
        theta: Rotation angle
        S: Skew-symmetric matrix
        L: Depth
        device: Device

    Returns:
        Theoretical OOD loss
    """
    D = Sigma.shape[0]

    # Rotate covariances
    Sigma_prime = rotate_covariance(Sigma, theta, S)
    Omega_prime = rotate_covariance(Omega, theta, S)

    # Compute Sigma^{-1}
    Sigma_inv = torch.linalg.inv(Sigma + 1e-6 * torch.eye(D, device=device))

    # Compute M = I - Sigma^{-1} @ Sigma'
    I = torch.eye(D, device=device)
    M = I - Sigma_inv @ Sigma_prime

    # Compute M^L
    M_L = torch.linalg.matrix_power(M, L)

    # Compute loss: tr(Omega' @ M_L^T @ Sigma' @ M_L)
    loss = torch.trace(Omega_prime @ M_L.T @ Sigma_prime @ M_L)

    return loss.item()


def run_figure_3c_theoretical(
    D: int = 64,
    nu: float = 1.0,
    beta: float = 1.0,
    device: str = 'cpu',
    save_path: str = None
):
    """
    Compute theoretical Figure 3(c) using Result 5.

    This computes the theoretical curves without training.
    """
    print("=" * 60)
    print("Figure 3(c): Brittleness to Distribution Shift (Theoretical)")
    print("=" * 60)

    # Create covariances
    Sigma = create_powerlaw_covariance(D, nu=nu, device=device)
    Omega = create_powerlaw_task_covariance(D, beta=beta, nu=nu, device=device)
    S = create_random_skew_symmetric(D, device=device, seed=123)

    depths = [1, 2, 4, 8]
    thetas = np.linspace(0, 0.25, 50)

    # Compute theoretical loss for each depth and theta
    results = {L: [] for L in depths}

    for theta in tqdm(thetas, desc="Computing theory"):
        for L in depths:
            loss = compute_theoretical_ood_loss(Sigma, Omega, theta, S, L, device)
            results[L].append(loss)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {1: '#1f1f1f', 2: '#8b008b', 4: '#b22222', 8: '#ff8c00'}

    for L in depths:
        ax.plot(thetas, results[L], color=colors[L], label=f'L = {L}', linewidth=2)

    ax.set_xlabel(r'$\theta$', fontsize=14)
    ax.set_ylabel(r'$\mathcal{L}_{OOD}$', fontsize=14)
    ax.set_title('(c) Brittleness to Distribution Shift (Theoretical)', fontsize=12)
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.25)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()

    return results


def run_figure_3c_comparison(
    D: int = 32,
    n_train_steps: int = 3000,
    lr: float = 0.0005,
    nu: float = 1.0,
    beta: float = 1.0,
    alpha: float = 4.0,
    device: str = 'cpu',
    save_path: str = None
):
    """
    Run both theoretical and experimental versions and plot side by side.

    This helps validate that experimental training produces results
    consistent with theory.
    """
    print("=" * 60)
    print("Figure 3(c): Comparison - Theoretical vs Experimental")
    print("=" * 60)

    # Create covariances
    Sigma = create_powerlaw_covariance(D, nu=nu, device=device)
    Omega = create_powerlaw_task_covariance(D, beta=beta, nu=nu, device=device)
    S = create_random_skew_symmetric(D, device=device, seed=123)

    depths = [1, 2, 4, 8]
    thetas = np.linspace(0, 0.25, 30)

    # --- Theoretical Results ---
    print("\nComputing theoretical results...")
    theoretical_results = {L: [] for L in depths}
    for theta in tqdm(thetas, desc="Theory"):
        for L in depths:
            loss = compute_theoretical_ood_loss(Sigma, Omega, theta, S, L, device)
            theoretical_results[L].append(loss)

    # --- Experimental Results ---
    print("\nTraining models for experimental results...")
    trained_models = {}
    for L in depths:
        print(f"\n  Training L={L}...")
        model, steps, losses = train_fs_model(
            D=D, L=L, Sigma=Sigma, Omega=Omega,
            n_steps=n_train_steps, lr=lr, alpha=alpha,
            eval_every=100, n_eval_contexts=50,
            device=device, seed=42, verbose=True,
            init_at_optimal=True
        )
        trained_models[L] = model
        print(f"    Final loss: {losses[-1]:.6f}")

    print("\nEvaluating experimental OOD loss...")
    experimental_results = {L: [] for L in depths}
    for theta in tqdm(thetas, desc="Experimental"):
        Sigma_prime = rotate_covariance(Sigma, theta, S)
        Omega_prime = rotate_covariance(Omega, theta, S)
        for L in depths:
            ood_loss = evaluate_ood_loss(
                trained_models[L], Sigma_prime, Omega_prime,
                D=D, alpha=alpha, n_contexts=100, device=device
            )
            experimental_results[L].append(ood_loss)

    # --- Plot Comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {1: '#1f1f1f', 2: '#8b008b', 4: '#b22222', 8: '#ff8c00'}

    # Left: Theoretical
    ax = axes[0]
    for L in depths:
        ax.plot(thetas, theoretical_results[L], color=colors[L],
                label=f'L = {L}', linewidth=2)
    ax.set_xlabel(r'$\theta$', fontsize=14)
    ax.set_ylabel(r'$\mathcal{L}_{OOD}$', fontsize=14)
    ax.set_title('Theoretical (Result 5)', fontsize=12)
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.25)

    # Right: Experimental
    ax = axes[1]
    for L in depths:
        ax.plot(thetas, experimental_results[L], color=colors[L],
                label=f'L = {L}', linewidth=2)
    ax.set_xlabel(r'$\theta$', fontsize=14)
    ax.set_ylabel(r'$\mathcal{L}_{OOD}$', fontsize=14)
    ax.set_title('Experimental (Trained Models)', fontsize=12)
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.25)

    plt.suptitle('Figure 3(c): Brittleness to Distribution Shift', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved to {save_path}")
    else:
        plt.show()

    plt.close()

    return theoretical_results, experimental_results


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

    # Run EXPERIMENTAL version (actual training)
    print("\n" + "=" * 60)
    print("Running EXPERIMENTAL Figure 3c (with actual training)")
    print("This will train models and evaluate on rotated covariances")
    print("=" * 60 + "\n")

    # Parameters chosen for proper convergence:
    # - init_at_optimal=True: Initialize Gamma near L*Sigma^{-1} for fast convergence
    # - alpha=4.0: Higher context ratio for better convergence
    # - n_train_steps=3000: Enough steps to fine-tune from optimal init
    # - lr=0.0005: Lower learning rate for stability near optimum
    ood_results, training_curves = run_figure_3c_experimental(
        D=32,
        n_train_steps=8000,
        lr=0.0005,
        nu=1.0,
        beta=1.0,
        alpha=4.0,
        device=device,
        save_path='results/figure_3c_experimental.png',
        init_at_optimal=True  # Initialize Gamma = L*Sigma^{-1} + noise
    )

    print("\nExperimental Figure 3c complete!")
    print("Results saved to results/figure_3c_experimental.png")

    # Print summary of results
    print("\n" + "=" * 60)
    print("Summary of OOD Loss Results:")
    print("=" * 60)
    print("theta=0 (in-distribution) losses:")
    for L in [1, 2, 4, 8]:
        print(f"  L={L}: {ood_results[L][0]:.6f}")
    print("\ntheta=0.25 (max rotation) losses:")
    for L in [1, 2, 4, 8]:
        print(f"  L={L}: {ood_results[L][-1]:.6f}")
    print("\nExpected behavior: Higher L should show FASTER growth (steeper curves)")
    print("This demonstrates the brittleness phenomenon from the paper.")

    # Also run comparison mode
    print("\n" + "=" * 60)
    print("Now running COMPARISON mode (Theory vs Experimental)")
    print("=" * 60 + "\n")

    theoretical_results, experimental_results = run_figure_3c_comparison(
        D=32,
        n_train_steps=3000,
        lr=0.0005,
        nu=1.0,
        beta=1.0,
        alpha=4.0,
        device=device,
        save_path='results/figure_3c_comparison.png'
    )

    print("\nAll done! Check results/ folder for:")
    print("  - figure_3c_experimental.png: Experimental results only")
    print("  - figure_3c_experimental_training.png: Training curves")
    print("  - figure_3c_comparison.png: Theory vs Experimental side-by-side")
