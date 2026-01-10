"""
Training script for RRS (Randomly Rotated Structured) setting experiments (Task 4).

Task 4: Extension - Robustness of RRS Setting

Key Idea (from paper Section 3.3, Result 6):
- FS model: Trains on FIXED covariance -> learns Gamma = L*Sigma^{-1} (full matrix) -> BRITTLE
- RRS model: Trains on RANDOMLY ROTATED covariances -> learns Gamma = gamma*I (SCALAR) -> ROBUST

Critical insight from paper:
"By introducing the random rotation across contexts, the model cannot encode a whitening
transform of the data in the matrix Gamma" -> forces isotropic solution Gamma = gamma * I

Experiment:
1. Train RRS models with SCALAR gamma (not full matrix!)
2. Test on same shifted covariances as Task 3
3. Compare: FS (brittle, steep curves) vs RRS (robust, flat curves)

Hypothesis: RRS model's OOD loss should remain flat as theta varies.
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

# Import from train_fs.py
from train_fs import (
    ReducedGammaModelFS,
    create_powerlaw_covariance,
    create_powerlaw_task_covariance,
    create_random_skew_symmetric,
    rotate_covariance,
    compute_icl_loss,
    train_fs_model,
    evaluate_ood_loss,
    generate_fs_data
)


class ReducedGammaModelRRS(nn.Module):
    """
    Reduced Gamma Model for RRS (Randomly Rotated Structured) setting.

    CRITICAL: Unlike FS where Gamma is a full DxD matrix,
    RRS forces Gamma = gamma * I (scalar times identity).

    From paper Result 6:
    "Gradient flow on the Gamma-reduced model maintains the isotropy condition Gamma(t) = gamma(t)*I"

    This is because random rotations prevent the model from learning any specific
    covariance structure - it must learn a rotation-invariant solution.

    NOTE ON IMPLEMENTATION CHOICE (from review):
    This implementation enforces scalar Γ=γI from the START (single parameter).
    This matches Result 6's CONSEQUENCE (that RRS converges to isotropic solution),
    but does not explicitly demonstrate that a full-matrix Γ would isotropize
    under random rotations during training.

    To fully demonstrate Result 6's dynamics, one could:
    1. Start with full D×D Γ matrix initialized randomly
    2. Train under RRS (random rotations each context)
    3. Observe Γ converging toward γI form empirically

    The current approach is sufficient for Task 4's goal (comparing FS vs RRS
    robustness) since it implements the converged behavior correctly.
    """

    def __init__(self, D: int, L: int = 1, init_gamma: float = 0.1):
        """
        Args:
            D: Dimension
            L: Depth
            init_gamma: Initial value for scalar gamma
        """
        super().__init__()
        self.D = D
        self.L = L
        # KEY: Only a SCALAR gamma parameter, not a full DxD matrix!
        self.gamma = nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32))

    def forward(self, X: torch.Tensor, y: torch.Tensor, X_star: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Gamma = gamma * I (isotropic).

        f(x*) = (1/LP) x*^T (gamma*I) Sum_{l=0}^{L-1} (I - L^{-1} Sigma_hat * gamma)^l X^T y

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

        # For RRS: Gamma = gamma * I
        # Compute M = I - (gamma/L) * Sigma_hat
        I = torch.eye(D, device=X.device, dtype=X.dtype)
        M = I - (self.gamma / L) * Sigma_hat

        # Compute geometric series: S = Sum_{l=0}^{L-1} M^l
        S = torch.zeros_like(I)
        M_power = I.clone()
        for ell in range(L):
            S = S + M_power
            if ell < L - 1:
                M_power = M_power @ M

        # Compute X^T y
        Xy = X.T @ y  # (D,)

        # Compute gamma * S @ X^T y (since Gamma = gamma * I)
        Gamma_S_Xy = self.gamma * (S @ Xy)  # (D,)

        # Compute predictions: f(x*) = (1/LP) x*^T Gamma S X^T y
        predictions = (X_star @ Gamma_S_Xy) / (L * P)  # (K,)

        return predictions


def generate_rrs_data(
    D: int,
    P: int,
    K: int,
    Sigma_base: torch.Tensor,
    Omega_base: torch.Tensor,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate data for the RRS (Randomly Rotated Structured) setting.

    Key difference from FS: Each context gets a RANDOM rotation of the base covariance.
    This forces the model to learn a rotation-invariant algorithm.

    Sigma_context = R @ Sigma_base @ R^T where R is random orthogonal

    Args:
        D: Dimension
        P: Number of training points
        K: Number of test points
        Sigma_base: Base data covariance (D, D) - will be randomly rotated
        Omega_base: Base task covariance (D, D) - will be randomly rotated
        device: Device

    Returns:
        X, y, X_star, y_star, beta
    """
    # Generate a random orthogonal matrix R (rotation)
    # Use QR decomposition of a random matrix
    # For MPS device, QR is not implemented, so move to CPU temporarily
    device_type = device if isinstance(device, str) else device.type
    if device_type == 'mps':
        random_matrix = torch.randn(D, D, device='cpu')
        Q, R_qr = torch.linalg.qr(random_matrix)
        # Ensure it's a proper rotation (det = +1)
        Q = Q * torch.sign(torch.diag(R_qr)).unsqueeze(0)
        # Move back to device
        Q = Q.to(device)
    else:
        random_matrix = torch.randn(D, D, device=device)
        Q, R_qr = torch.linalg.qr(random_matrix)
        # Ensure it's a proper rotation (det = +1)
        Q = Q * torch.sign(torch.diag(R_qr)).unsqueeze(0)

    # Rotate covariances: Sigma' = Q @ Sigma @ Q^T
    Sigma_rotated = Q @ Sigma_base @ Q.T
    Omega_rotated = Q @ Omega_base @ Q.T

    # Sample task vector beta ~ N(0, Omega_rotated)
    Omega_sqrt = torch.linalg.cholesky(Omega_rotated + 1e-6 * torch.eye(D, device=device))
    z = torch.randn(D, device=device)
    beta = Omega_sqrt @ z

    # Sample training inputs x ~ N(0, Sigma_rotated)
    Sigma_sqrt = torch.linalg.cholesky(Sigma_rotated + 1e-6 * torch.eye(D, device=device))
    Z_train = torch.randn(P, D, device=device)
    X = Z_train @ Sigma_sqrt.T

    # Sample test inputs
    Z_test = torch.randn(K, D, device=device)
    X_star = Z_test @ Sigma_sqrt.T

    # Compute targets: y = (1/sqrt(D)) beta . x
    y = (X @ beta) / np.sqrt(D)
    y_star = (X_star @ beta) / np.sqrt(D)

    return X, y, X_star, y_star, beta


def train_rrs_model(
    D: int,
    L: int,
    Sigma_base: torch.Tensor,
    Omega_base: torch.Tensor,
    n_steps: int = 5000,
    lr: float = 0.001,
    alpha: float = 2.0,
    eval_every: int = 100,
    n_eval_contexts: int = 50,
    device: str = 'cpu',
    seed: int = 42,
    verbose: bool = True
) -> Tuple[ReducedGammaModelRRS, List[float], List[float]]:
    """
    Train ReducedGammaModelRRS (SCALAR gamma) on RRS data.

    CRITICAL: Uses scalar gamma model, NOT full matrix model!

    From paper Result 6:
    "Gradient flow on the Gamma-reduced model maintains the isotropy condition Gamma(t) = gamma(t)*I"

    Key difference from FS training:
    - Each training context uses a RANDOMLY ROTATED covariance
    - Model has only 1 parameter (scalar gamma), not D*D parameters
    - Model must learn rotation-invariant algorithm
    - At convergence: gamma -> optimal scalar for isotropic predictor

    Args:
        D: Dimension
        L: Depth
        Sigma_base: Base data covariance (will be randomly rotated each context)
        Omega_base: Base task covariance
        n_steps: Training steps
        lr: Learning rate
        alpha: Context length ratio P/D
        eval_every: Evaluation frequency
        n_eval_contexts: Contexts for evaluation
        device: Device
        seed: Random seed
        verbose: Print progress

    Returns:
        model: Trained ReducedGammaModelRRS (scalar gamma)
        steps: List of step numbers
        losses: List of losses
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    P = max(1, int(alpha * D))
    K = max(1, int(alpha * D))

    Sigma_base = Sigma_base.to(device)
    Omega_base = Omega_base.to(device)

    # CRITICAL: Use ReducedGammaModelRRS with SCALAR gamma, not full matrix!
    # Initialize gamma near L (optimal for isotropic case per paper Eq 47)
    init_gamma = float(L) * 0.5  # Start at 0.5*L, will converge toward optimal
    model = ReducedGammaModelRRS(D=D, L=L, init_gamma=init_gamma).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    steps = []
    losses = []

    iterator = tqdm(range(n_steps), desc=f"RRS L={L}") if verbose else range(n_steps)

    for step in iterator:
        # Generate RRS data (random rotation each step)
        X, y, X_star, y_star, _ = generate_rrs_data(D, P, K, Sigma_base, Omega_base, device)

        # Training step
        optimizer.zero_grad()
        loss = compute_icl_loss(model, X, y, X_star, y_star)

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Evaluate
        if step % eval_every == 0:
            model.eval()
            eval_loss = 0.0

            with torch.no_grad():
                for _ in range(n_eval_contexts):
                    X, y, X_star, y_star, _ = generate_rrs_data(
                        D, P, K, Sigma_base, Omega_base, device
                    )
                    loss_val = compute_icl_loss(model, X, y, X_star, y_star)
                    eval_loss += loss_val.item()

            eval_loss /= n_eval_contexts
            steps.append(step)
            losses.append(eval_loss)

            if verbose:
                # Show gamma value for debugging - it should converge to optimal
                gamma_val = model.gamma.item()
                iterator.set_postfix(loss=f"{eval_loss:.4f}", gamma=f"{gamma_val:.3f}")

            model.train()

    # Print final gamma value
    if verbose:
        print(f"    Final gamma = {model.gamma.item():.4f} (optimal for isotropic ~ {float(L):.1f})")

    return model, steps, losses


def run_task4_comparison(
    D: int = 32,
    n_train_steps_fs: int = 3000,
    n_train_steps_rrs: int = 5000,
    lr: float = 0.001,
    nu: float = 1.0,
    beta_param: float = 1.0,
    alpha: float = 4.0,
    device: str = 'cpu',
    save_path: str = None
):
    """
    Task 4: Compare FS vs RRS robustness to distribution shift.

    This is your novel extension from the proposal.

    Experiment:
    1. Train FS models on fixed Sigma (from Task 3)
    2. Train RRS models on randomly rotated Sigma
    3. Evaluate BOTH on the same shifted covariances (varying theta)
    4. Compare: FS should be brittle, RRS should be robust

    Args:
        D: Dimension
        n_train_steps_fs: Training steps for FS models
        n_train_steps_rrs: Training steps for RRS models (needs more since harder)
        lr: Learning rate
        nu: Power-law exponent
        beta_param: Source exponent
        alpha: Context length ratio
        device: Device
        save_path: Path to save figure
    """
    print("=" * 70)
    print("Task 4: Extension - FS vs RRS Robustness Comparison")
    print("=" * 70)
    print("\nHypothesis: RRS model should be ROBUST (flat curves)")
    print("            FS model should be BRITTLE (steep curves)")
    print("=" * 70)

    # Create base covariance matrices
    Sigma_base = create_powerlaw_covariance(D, nu=nu, device=device)
    Omega_base = create_powerlaw_task_covariance(D, beta=beta_param, nu=nu, device=device)

    # Create skew-symmetric matrix for test rotations
    S = create_random_skew_symmetric(D, device=device, seed=123)

    # Depths to test
    depths = [1, 2, 4, 8]

    # Rotation angles for evaluation
    thetas = np.linspace(0, 0.25, 25)

    # ========== Phase 1: Train FS models ==========
    print("\n" + "=" * 50)
    print("Phase 1: Training FS models (Fixed Covariance)")
    print("=" * 50)

    fs_models = {}
    fs_training_curves = {}

    for L in depths:
        print(f"\n  Training FS L={L}...")
        model, steps, losses = train_fs_model(
            D=D, L=L, Sigma=Sigma_base, Omega=Omega_base,
            n_steps=n_train_steps_fs, lr=lr/2, alpha=alpha,
            eval_every=100, n_eval_contexts=50,
            device=device, seed=42, verbose=True,
            init_at_optimal=True
        )
        fs_models[L] = model
        fs_training_curves[L] = (steps, losses)
        print(f"    Final FS loss: {losses[-1]:.6f}")

    # ========== Phase 2: Train RRS models ==========
    print("\n" + "=" * 50)
    print("Phase 2: Training RRS models (Random Rotations)")
    print("=" * 50)

    rrs_models = {}
    rrs_training_curves = {}

    for L in depths:
        print(f"\n  Training RRS L={L}...")
        model, steps, losses = train_rrs_model(
            D=D, L=L, Sigma_base=Sigma_base, Omega_base=Omega_base,
            n_steps=n_train_steps_rrs, lr=lr, alpha=alpha,
            eval_every=100, n_eval_contexts=50,
            device=device, seed=42, verbose=True
        )
        rrs_models[L] = model
        rrs_training_curves[L] = (steps, losses)
        print(f"    Final RRS loss: {losses[-1]:.6f}")

    # ========== Phase 3: Evaluate on shifted covariances ==========
    print("\n" + "=" * 50)
    print("Phase 3: Evaluating on shifted covariances")
    print("=" * 50)

    fs_ood_results = {L: [] for L in depths}
    rrs_ood_results = {L: [] for L in depths}

    P = max(1, int(alpha * D))
    K = max(1, int(alpha * D))

    for theta in tqdm(thetas, desc="Evaluating OOD"):
        # Rotate base covariance by theta
        Sigma_shifted = rotate_covariance(Sigma_base, theta, S)
        # Align with paper's distribution-shift setup: rotate Σ only, keep Ω fixed.
        Omega_shifted = Omega_base

        for L in depths:
            # Evaluate FS model
            fs_loss = evaluate_ood_loss(
                fs_models[L], Sigma_shifted, Omega_shifted,
                D=D, alpha=alpha, n_contexts=100, device=device
            )
            fs_ood_results[L].append(fs_loss)

            # Evaluate RRS model
            rrs_loss = evaluate_ood_loss(
                rrs_models[L], Sigma_shifted, Omega_shifted,
                D=D, alpha=alpha, n_contexts=100, device=device
            )
            rrs_ood_results[L].append(rrs_loss)

    # ========== Phase 4: Create comparison plot ==========
    print("\n" + "=" * 50)
    print("Phase 4: Creating comparison plot")
    print("=" * 50)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {1: '#1f1f1f', 2: '#8b008b', 4: '#b22222', 8: '#ff8c00'}

    # Left: FS models (should show brittleness)
    ax = axes[0]
    for L in depths:
        ax.plot(thetas, fs_ood_results[L], color=colors[L],
                label=f'L = {L}', linewidth=2)
    ax.set_xlabel(r'$\theta$ (rotation angle)', fontsize=12)
    ax.set_ylabel(r'$\mathcal{L}_{OOD}$', fontsize=12)
    ax.set_title('FS Model (Fixed Covariance)\nBRITTLE - learns preconditioner', fontsize=11)
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.25)

    # Right: RRS models (should show robustness)
    ax = axes[1]
    for L in depths:
        ax.plot(thetas, rrs_ood_results[L], color=colors[L],
                label=f'L = {L}', linewidth=2)
    ax.set_xlabel(r'$\theta$ (rotation angle)', fontsize=12)
    ax.set_ylabel(r'$\mathcal{L}_{OOD}$', fontsize=12)
    ax.set_title('RRS Model (Random Rotations)\nROBUST - learns general algorithm', fontsize=11)
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.25)

    plt.suptitle('Task 4: FS vs RRS Robustness to Distribution Shift', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved to {save_path}")
    else:
        plt.show()

    plt.close()

    # ========== Also create training curves comparison ==========
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    # FS training curves
    ax = axes2[0]
    for L in depths:
        steps, losses = fs_training_curves[L]
        ax.plot(steps, losses, color=colors[L], label=f'L = {L}', linewidth=1.5)
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('FS Training Curves', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RRS training curves
    ax = axes2[1]
    for L in depths:
        steps, losses = rrs_training_curves[L]
        ax.plot(steps, losses, color=colors[L], label=f'L = {L}', linewidth=1.5)
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('RRS Training Curves', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Training Curves: FS vs RRS', fontsize=12)
    plt.tight_layout()

    if save_path:
        train_path = save_path.replace('.png', '_training.png')
        plt.savefig(train_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {train_path}")

    plt.close()

    # ========== Print summary ==========
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\nFS Model OOD Loss (theta=0 vs theta=0.25):")
    for L in depths:
        ratio = fs_ood_results[L][-1] / fs_ood_results[L][0]
        print(f"  L={L}: {fs_ood_results[L][0]:.4f} -> {fs_ood_results[L][-1]:.4f} (ratio: {ratio:.1f}x)")

    print("\nRRS Model OOD Loss (theta=0 vs theta=0.25):")
    for L in depths:
        ratio = rrs_ood_results[L][-1] / rrs_ood_results[L][0]
        gamma_val = rrs_models[L].gamma.item()
        print(f"  L={L}: {rrs_ood_results[L][0]:.4f} -> {rrs_ood_results[L][-1]:.4f} (ratio: {ratio:.1f}x, gamma={gamma_val:.3f})")

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("- If FS ratios are HIGH (e.g., 100x+) -> FS is BRITTLE (confirms Task 3)")
    print("- If RRS ratios are LOW (e.g., <10x) -> RRS is ROBUST (confirms hypothesis)")
    print("- This proves RRS learns general ICL algorithm, not just a preconditioner")
    print("=" * 70)

    return fs_ood_results, rrs_ood_results, fs_training_curves, rrs_training_curves


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

    # Run Task 4 comparison
    print("\n" + "=" * 70)
    print("TASK 4: Extension - Robustness of RRS Setting")
    print("=" * 70)
    print("\nThis experiment compares:")
    print("  - FS (Fixed Structured): Trains on fixed covariance")
    print("  - RRS (Randomly Rotated Structured): Trains on random rotations")
    print("\nBoth are tested on the SAME shifted covariances from Task 3")
    print("=" * 70 + "\n")

    fs_results, rrs_results, fs_curves, rrs_curves = run_task4_comparison(
        D=32,
        n_train_steps_fs=3000,
        n_train_steps_rrs=5000,  # RRS needs more steps (harder problem)
        lr=0.001,
        nu=1.0,
        beta_param=1.0,
        alpha=4.0,
        device=device,
        save_path='results/task4_fs_vs_rrs.png'
    )

    print("\nTask 4 complete!")
    print("Results saved to:")
    print("  - results/task4_fs_vs_rrs.png (main comparison)")
    print("  - results/task4_fs_vs_rrs_training.png (training curves)")
