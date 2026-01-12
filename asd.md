Your nonlinear data y=ReLU(β·x/√D)+σε is consistent with “paper ISO + ReLU before noise” (train_nonlinear.py (line 114)).
The model is not the literal “add a 2-layer readout on top of the paper’s h^L” (F=aᵀ ReLU(W h^L)): you instead define the feature going into the MLP as φ(x*) = x* ⊙ rep (train_nonlinear.py (line 37)). That’s a reasonable engineered extension, but it’s not the same object as the paper’s residual-stream h^L, so don’t describe it as a direct architectural conversion unless you justify that identification.
Also: ReLU(z) has positive mean, but neither your linear baseline nor your 2-layer head has a bias term; this biases the comparison (and can explain the linear model flattening around a nontrivial MSE in task5_linear_vs_2layer.png). For a cleaner comparison, either center labels (e.g., ReLU(z) - E[ReLU(N(0,1))]) or add bias(es).











Data generation (lines 156-162): Now computes β-conditioned centering:


s = torch.norm(beta) / np.sqrt(D)
mean_relu = s / np.sqrt(2 * np.pi)
y = F.relu(z_train) - mean_relu
Docstrings and titles: Updated to reflect centered data

This fix:

Preserves the paper's "through-origin" regression style (zero-mean targets, no bias needed)
Is exact for any D (not relying on large-D approximation)
Makes the comparison between linear and 2-layer models cleaner
