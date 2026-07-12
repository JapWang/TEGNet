import numpy as np
import pandas as pd

class OccupancyGating:
    """
    TEG-Net: Occupancy Gating module (Methodology Section 3.4).
    Uses Hill-function saturation kinetics to project the macroscopic physical network
    W_global down to single-cell resolution, generating a 3D cell-specific micro-causal
    tensor J (N cells x M targets x M sources).
    """

    def __init__(self, n_hill: float = 2.0):
        # Hill coefficient — n=2 reflects typical dimerization binding kinetics
        self.n_hill = n_hill
        self.K_activation_ = None  # Adaptive half-max activation constants per gene
        self.J_tensor_ = None      # Final 3D causal tensor

    def fit_transform(self, df_X: pd.DataFrame, df_W_global: pd.DataFrame) -> np.ndarray:
        """
        Micro-tensor assembly pipeline.
        :param df_X: Static normalized expression matrix (N cells x M genes)
        :param df_W_global: Macro physical network W_global (M x M, row=target, col=source)
        :return: 3D cell-specific causal tensor J (N x M x M)
        """
        X = df_X.values
        W_global = df_W_global.values
        n_samples, n_features = X.shape

        print("\n" + "="*60)
        print("  [TEG-Net Micro-Engine] Assembling cell-specific occupancy-gated tensor...")
        print("="*60)

        # Step 1: Extract adaptive activation constant K_k per source gene
        K_activation = np.zeros(n_features)

        for k in range(n_features):
            X_k = X[:, k]
            non_zero_vals = X_k[X_k > 1e-8]

            if len(non_zero_vals) > 0:
                k_val = np.median(non_zero_vals)
                if k_val < 1e-8:
                    k_val = np.min(non_zero_vals)
            else:
                k_val = 1e-8

            K_activation[k] = k_val

        self.K_activation_ = K_activation

        # Step 2: Hill-function nonlinear occupancy gating
        print(f"  -> Applying Hill saturation kinetics (n={self.n_hill})...")

        # Pre-allocate 3D tensor: [samples N, target gene j, source gene k]
        J_tensor = np.zeros((n_samples, n_features, n_features))

        # Vectorized assembly via NumPy broadcasting
        for k in range(n_features):
            W_col_k = W_global[:, k]

            if np.all(np.abs(W_col_k) < 1e-12):
                continue

            X_all_k = X[:, k]

            # Hill function: g_k = X_k^n / (K_k^n + X_k^n)
            X_k_n = np.power(np.clip(X_all_k, 0, None), self.n_hill)
            K_k_n = np.power(self.K_activation_[k], self.n_hill)

            g_k = X_k_n / (K_k_n + X_k_n + 1e-16)

            # Broadcast: cell-specific flux = g_k (outer) W_col_k
            J_tensor[:, :, k] = np.outer(g_k, W_col_k)

        self.J_tensor_ = J_tensor

        print(f"  [+] Assembly complete. Output micro-causal tensor shape: {J_tensor.shape}")
        print("="*60 + "\n")

        return J_tensor
