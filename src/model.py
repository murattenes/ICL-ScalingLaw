"""
Reduced Linear Attention Model for In-Context Learning
Based on: "Theory of Scaling Laws for In-Context Regression" by Bordelon et al. (2025)

This implements the Reduced Gamma Model (Equation 4) and the gradient flow dynamics
for in-context linear regression.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class ReducedGammaModel(nn.Module):
    """
    Reduced Gamma Model for In-Context Learning (Equation 4 from the paper).

    For ISO setting, Γ = γI where γ is a scalar parameter.
    The predictor is: f(x*) = (1/LP) x*^T Γ Σ_{ℓ=0}^{L-1} (I - L^{-1} Σ̂ Γ)^ℓ X^T y

    where Σ̂ = (1/P) X X^T is the empirical covariance.
    """

    def __init__(self, D: int, L: int = 1, init_gamma: float = 0.0):
        """
        Args:
            D: Input dimension
            L: Depth (number of layers/steps)
            init_gamma: Initial value for gamma parameter
        """
        super().__init__()
        self.D = D
        self.L = L
        # For ISO setting, Γ = γI, so we only need scalar γ
        self.gamma = nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32))

    def compute_empirical_covariance(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute empirical covariance Σ̂ = (1/P) X^T X

        Args:
            X: Input data of shape (P, D)
        Returns:
            Σ̂ of shape (D, D)
        """
        P = X.shape[0]
        return (X.T @ X) / P  # (D, D)

    def forward(self, X: torch.Tensor, y: torch.Tensor, X_star: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing Equation 4.

        f(x*) = (1/LP) x*^T Γ Σ_{ℓ=0}^{L-1} (I - L^{-1} Σ̂ Γ)^ℓ X^T y

        Args:
            X: Training inputs of shape (P, D)
            y: Training targets of shape (P,) or (P, 1)
            X_star: Test inputs of shape (K, D)

        Returns:
            Predictions of shape (K,)
        """
        P, D = X.shape
        L = self.L

        # Ensure y is shape (P,)
        y = y.squeeze()

        # Compute empirical covariance: Σ̂ = (1/P) X^T X (D x D)
        Sigma_hat = self.compute_empirical_covariance(X)

        # For ISO: Γ = γI
        # Compute the sum: Σ_{ℓ=0}^{L-1} (I - L^{-1} Σ̂ Γ)^ℓ
        # = Σ_{ℓ=0}^{L-1} (I - L^{-1} γ Σ̂)^ℓ

        I = torch.eye(D, device=X.device, dtype=X.dtype)
        M = I - (self.gamma / L) * Sigma_hat  # (I - L^{-1} γ Σ̂)

        # Compute the geometric series sum
        # S = I + M + M^2 + ... + M^{L-1}
        S = torch.zeros_like(I)
        M_power = I.clone()
        for ell in range(L):
            S = S + M_power
            if ell < L - 1:
                M_power = M_power @ M

        # Compute X^T y (D,)
        Xy = X.T @ y

        # Compute Γ S X^T y = γ S X^T y (since Γ = γI)
        Gamma_S_Xy = self.gamma * (S @ Xy)  # (D,)

        # Compute predictions: f(x*) = (1/LP) x*^T Γ S X^T y
        predictions = (X_star @ Gamma_S_Xy) / (L * P)  # (K,)

        return predictions

    def compute_residual(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the residual stream dynamics on training points.

        Δ^ℓ = (I - (1/LP) X^T Γ X)^ℓ y (Equation 51)

        For ISO: Δ^L = (I - (γ/LP) X^T X)^L y
        """
        P, D = X.shape
        y = y.squeeze()

        # Compute X^T X / P = Σ̂ (in sample covariance space)
        # But Equation 51 uses X^T Γ X / (LP)
        # For Γ = γI: (1/LP) X^T γI X = (γ/LP) X^T X

        # We work in the P-dimensional space here
        # A = (γ/(LP)) X X^T which is P x P
        A = (self.gamma / (self.L * P)) * (X @ X.T)  # (P, P)

        I_P = torch.eye(P, device=X.device, dtype=X.dtype)
        M = I_P - A

        # Compute M^L y
        Delta = y.clone()
        for _ in range(self.L):
            Delta = M @ Delta

        return Delta


class LinearAttentionICL(nn.Module):
    """
    Full Linear Attention model for ICL (Equation 3 from the paper).

    This implements the full attention mechanism with separate W_x, W_k, W_q, W_v matrices.
    """

    def __init__(self, D: int, L: int = 1, hidden_dim: Optional[int] = None):
        """
        Args:
            D: Input dimension
            L: Depth (number of layers)
            hidden_dim: Hidden dimension for residual stream (default: D+1)
        """
        super().__init__()
        self.D = D
        self.L = L
        self.hidden_dim = hidden_dim or (D + 1)

        # Initialize weights according to the paper's alignment conditions
        # W_x: maps x to residual stream (D -> hidden_dim)
        # w_y: maps y to residual stream (1 -> hidden_dim)
        # w_o: readout weights (hidden_dim -> 1)

        self.W_x = nn.Parameter(torch.randn(self.hidden_dim, D) * 0.01)
        self.w_y = nn.Parameter(torch.randn(self.hidden_dim) * 0.01)
        self.w_o = nn.Parameter(torch.randn(self.hidden_dim) * 0.01)

        # Attention weights (shared across layers for looped/universal transformer)
        self.W_k = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim) * 0.01)
        self.W_q = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim) * 0.01)
        self.W_v = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim) * 0.01)

    def forward(self, X: torch.Tensor, y: torch.Tensor, X_star: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing Equation 3.

        Args:
            X: Training inputs (P, D)
            y: Training targets (P,)
            X_star: Test inputs (K, D)

        Returns:
            Predictions (K,)
        """
        P, D = X.shape
        K = X_star.shape[0]
        y = y.squeeze()

        # Combine training and test points
        # For training points: h^1 = W_x @ x + w_y * y
        # For test points: h^1 = W_x @ x (no y provided)

        # Initialize residual stream for training points
        H_train = self.W_x @ X.T + self.w_y.unsqueeze(1) * y.unsqueeze(0)  # (hidden, P)

        # Initialize residual stream for test points (y=0)
        H_test = self.W_x @ X_star.T  # (hidden, K)

        # Run through L layers
        for ell in range(self.L):
            # Compute keys, queries, values for training points
            K_train = self.W_k @ H_train  # (hidden, P)
            V_train = self.W_v @ H_train  # (hidden, P)

            # Update training points (they attend to each other with negative sign)
            Q_train = self.W_q @ H_train  # (hidden, P)
            attn_train = (K_train.T @ Q_train) / (self.L * P)  # (P, P)
            H_train = H_train - V_train @ attn_train  # Update with negative sign

            # Update test points (they attend to training points with positive sign)
            Q_test = self.W_q @ H_test  # (hidden, K)
            attn_test = (K_train.T @ Q_test) / (self.L * P)  # (P, K)
            H_test = H_test + V_train @ attn_test  # Update with positive sign

        # Readout
        predictions = self.w_o @ H_test  # (K,)

        return predictions


def generate_iso_data(D: int, P: int, K: int = 1, sigma: float = 0.0,
                      device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor,
                                                      torch.Tensor, torch.Tensor,
                                                      torch.Tensor]:
    """
    Generate data for the ISO (isotropic) setting (Equation 5).

    x_μ ~ N(0, I)
    β ~ N(0, I)
    y_μ = (1/√D) β · x_μ + σ ε_μ, ε ~ N(0, 1)

    Args:
        D: Input dimension
        P: Number of training points (context length)
        K: Number of test points
        sigma: Noise standard deviation
        device: Device to create tensors on

    Returns:
        X: Training inputs (P, D)
        y: Training targets (P,)
        X_star: Test inputs (K, D)
        y_star: Test targets (K,)
        beta: True weights (D,)
    """
    # Sample task vector β ~ N(0, I)
    beta = torch.randn(D, device=device)

    # Sample training inputs x ~ N(0, I)
    X = torch.randn(P, D, device=device)

    # Sample test inputs
    X_star = torch.randn(K, D, device=device)

    # Compute targets: y = (1/√D) β · x + σε
    y = (X @ beta) / np.sqrt(D)
    y_star = (X_star @ beta) / np.sqrt(D)

    # Add noise
    if sigma > 0:
        y = y + sigma * torch.randn(P, device=device)
        y_star = y_star + sigma * torch.randn(K, device=device)

    return X, y, X_star, y_star, beta


def compute_icl_loss(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                     X_star: torch.Tensor, y_star: torch.Tensor) -> torch.Tensor:
    """
    Compute the ICL loss: L = (1/K) Σ_{μ=P+1}^{P+K} (f_μ - y_μ)²

    Args:
        model: The ICL model
        X: Training inputs (P, D)
        y: Training targets (P,)
        X_star: Test inputs (K, D)
        y_star: Test targets (K,)

    Returns:
        MSE loss
    """
    predictions = model(X, y, X_star)
    loss = torch.mean((predictions - y_star) ** 2)
    return loss


def compute_population_loss_iso(gamma: float, L: int, alpha: float,
                                 sigma: float = 0.0) -> float:
    """
    Compute the theoretical population loss for the ISO setting.

    Uses the Marchenko-Pastur distribution for the eigenvalues of Σ̂.

    L = ∫ ρ(λ) (1 - L^{-1} γ λ)^{2L} dλ

    For ISO with large D, this can be approximated numerically.

    Args:
        gamma: The gamma parameter
        L: Depth
        alpha: P/D ratio (context length / dimension)
        sigma: Noise level

    Returns:
        Population loss
    """
    # Marchenko-Pastur distribution parameters
    # For x ~ N(0, I), Σ̂ = (1/P) X^T X has MP distribution
    # with λ_+ = (1 + 1/√α)² and λ_- = (1 - 1/√α)² if α >= 1

    # Numerical integration over MP density.
    n_samples = 10000
    lambda_minus = (1 - 1 / np.sqrt(alpha)) ** 2
    lambda_plus = (1 + 1 / np.sqrt(alpha)) ** 2

    # Use a change of variables to handle the integrable singularity near lambda_minus.
    t = np.linspace(0.0, 1.0, n_samples)
    lambdas = lambda_minus + (lambda_plus - lambda_minus) * (t**2)
    lambdas = np.maximum(lambdas, 1e-12)

    density = (alpha / (2 * np.pi)) * np.sqrt(
        np.maximum(0, (lambda_plus - lambdas) * (lambdas - lambda_minus))
    ) / lambdas

    dt = t[1] - t[0]
    dlambda_dt = 2 * (lambda_plus - lambda_minus) * t
    weights = density * dlambda_dt * dt

    residuals = (1 - gamma * lambdas / L) ** (2 * L)
    continuous_loss = np.sum(weights * residuals)

    if alpha < 1:
        mass_at_zero = 1 - alpha
        return mass_at_zero * 1.0 + continuous_loss

    return continuous_loss


def optimal_gamma_iso(L: int, alpha: float) -> float:
    """
    Find the optimal gamma for the ISO setting that minimizes the loss.

    For L=1: optimal γ* ≈ α/(1+α)
    For L→∞: optimal γ* → L (when α > 1)

    Args:
        L: Depth
        alpha: P/D ratio

    Returns:
        Optimal gamma value
    """
    from scipy.optimize import minimize_scalar

    def loss_fn(gamma):
        return compute_population_loss_iso(gamma, L, alpha)

    # Search in reasonable range
    result = minimize_scalar(loss_fn, bounds=(0, 2*L), method='bounded')
    return result.x


if __name__ == "__main__":
    # Quick test
    D = 32
    P = 32  # alpha = 1
    K = 10
    L = 4

    print("Testing ReducedGammaModel...")
    model = ReducedGammaModel(D=D, L=L, init_gamma=1.0)
    X, y, X_star, y_star, beta = generate_iso_data(D, P, K)

    preds = model(X, y, X_star)
    loss = compute_icl_loss(model, X, y, X_star, y_star)

    print(f"Predictions shape: {preds.shape}")
    print(f"Loss: {loss.item():.4f}")
    print("Test passed!")
