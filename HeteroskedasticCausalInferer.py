import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLasso, empirical_covariance
from sklearn.linear_model import RidgeCV, LinearRegression, LassoCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import pdist
from scipy.interpolate import UnivariateSpline
from sklearn.preprocessing import PolynomialFeatures
import warnings

class HeteroskedasticCausalInferer:
    """
    TEG-Net core causal polarity engine.
    Four-step pipeline:
      1. EBIC Graphical Lasso for undirected conditional independence skeleton.
      2. Bidirectional partial residual extraction (linear or polynomial).
      3. HSIC (Hilbert-Schmidt Independence Criterion) asymmetry quantification.
      4. Parameter-free HSIC-ratio polarity mapping to produce confidence matrix Pi.
    """

    def __init__(self, knn_k: int = 15, ebic_gamma: float = 0.5, poly_degree: int = 1):
        self.knn_k = knn_k
        self.ebic_gamma = ebic_gamma
        self.poly_degree = poly_degree
        self.skeleton_ = None
        self.hsic_matrix_ = None
        self.polarity_matrix_ = None

    def fit_transform(self, df_X: pd.DataFrame) -> pd.DataFrame:
        """
        Core inference pipeline — lossless kernel compatible with MinMax-scaled data.
        """
        X = df_X.values
        n_samples, n_features = X.shape
        gene_names = df_X.columns.tolist()

        print("\n" + "="*60)
        print("  [TEG-Net Polarity Engine] Starting heteroskedastic causal rupture analysis...")
        print("="*60)

        # Step 1: EBIC Graphical Lasso for undirected conditional independence skeleton
        print("  -> Step 1: EBIC Graphical Lasso extracting undirected skeleton...")

        np.random.seed(42)
        jitter = np.random.normal(0, 1e-6, size=X.shape)
        X_gl = X + jitter

        S = empirical_covariance(X_gl)

        # L1 penalty space adapted for MinMax-scaled data covariance
        alphas = np.logspace(-5, -1, 60)

        best_ebic = np.inf
        best_precision = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for alpha in alphas:
                gl = GraphicalLasso(alpha=alpha, max_iter=500)
                try:
                    gl.fit(X_gl)
                    Theta = gl.precision_

                    E = (np.sum(np.abs(Theta) > 1e-5) - n_features) / 2
                    if E <= 0: continue

                    sign, logdet = np.linalg.slogdet(Theta)
                    if sign <= 0: continue
                    log_lik = logdet - np.trace(S @ Theta)

                    ebic = -n_samples * log_lik + E * np.log(n_samples) + 4 * E * self.ebic_gamma * np.log(n_features)

                    if ebic < best_ebic:
                        best_ebic = ebic
                        best_precision = Theta
                except Exception:
                    continue

        if best_precision is None:
            raise ValueError("EBIC Graphical Lasso did not converge. Check input grid or data variance.")

        skeleton = np.abs(best_precision) > 1e-5
        np.fill_diagonal(skeleton, False)
        self.skeleton_ = skeleton

        n_edges = np.sum(skeleton) // 2
        print(f"     [+] EBIC selection complete. Extracted {n_edges} undirected conditional edges.")

        # Step 2 & 3: Bidirectional partial residual + HSIC quantification
        print("  -> Step 2 & 3: Computing bidirectional heteroskedastic HSIC (full-sample)...")

        hsic_matrix = np.zeros((n_features, n_features))
        total_edges = np.sum(skeleton) // 2
        edge_counter = 0

        for i in range(n_features):
            for j in range(i + 1, n_features):
                if not skeleton[i, j]: continue

                edge_counter += 1
                if edge_counter % 10 == 0 or edge_counter == total_edges:
                    print(f"     ... HSIC independence test: {edge_counter}/{total_edges} skeleton edges")

                adj_i = set(np.where(skeleton[i, :])[0])
                adj_j = set(np.where(skeleton[j, :])[0])
                Z_idx = list((adj_i | adj_j) - {i, j})

                if len(Z_idx) > 0:
                    Z = X[:, Z_idx]
                    if len(Z_idx) > 50:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            lasso_i = LassoCV(cv=3).fit(Z, X[:, i])
                            lasso_j = LassoCV(cv=3).fit(Z, X[:, j])
                            active_Z_i = Z[:, lasso_i.coef_ != 0]
                            active_Z_j = Z[:, lasso_j.coef_ != 0]
                    else:
                        active_Z_i = Z
                        active_Z_j = Z

                    # Nonlinear polynomial feature expansion
                    if self.poly_degree > 1:
                        poly = PolynomialFeatures(degree=self.poly_degree, include_bias=False)
                        Z_poly_i = poly.fit_transform(active_Z_i) if active_Z_i.shape[1] > 0 else active_Z_i
                        Z_poly_j = poly.fit_transform(active_Z_j) if active_Z_j.shape[1] > 0 else active_Z_j
                    else:
                        Z_poly_i = active_Z_i
                        Z_poly_j = active_Z_j

                    # RidgeCV for robust partial residual extraction
                    ridge_i = RidgeCV(cv=5).fit(Z_poly_i, X[:, i]) if Z_poly_i.shape[1] > 0 else None
                    ridge_j = RidgeCV(cv=5).fit(Z_poly_j, X[:, j]) if Z_poly_j.shape[1] > 0 else None

                    X_tilde_i = X[:, i] - ridge_i.predict(Z_poly_i) if ridge_i else X[:, i]
                    X_tilde_j = X[:, j] - ridge_j.predict(Z_poly_j) if ridge_j else X[:, j]
                else:
                    X_tilde_i = X[:, i].copy()
                    X_tilde_j = X[:, j].copy()

                r_norm_j = self._normalize_variance(X_tilde_i, X_tilde_j)
                hsic_ij = self._compute_hsic(X_tilde_i, r_norm_j)

                r_norm_i = self._normalize_variance(X_tilde_j, X_tilde_i)
                hsic_ji = self._compute_hsic(X_tilde_j, r_norm_i)

                hsic_matrix[i, j] = hsic_ij
                hsic_matrix[j, i] = hsic_ji

        self.hsic_matrix_ = hsic_matrix.copy()

        # Step 4: Parameter-free HSIC-ratio polarity mapping
        print("  -> Step 4: Adaptive HSIC-ratio polarity mapping (parameter-free)...")

        Pi_matrix = np.zeros((n_features, n_features))

        for i in range(n_features):
            for j in range(i + 1, n_features):
                if skeleton[i, j]:
                    # Lower HSIC → more independent → causal source
                    sum_hsic = hsic_matrix[i, j] + hsic_matrix[j, i]
                    if sum_hsic > 1e-16:
                        Pi_matrix[j, i] = hsic_matrix[j, i] / sum_hsic
                        Pi_matrix[i, j] = hsic_matrix[i, j] / sum_hsic
                    else:
                        Pi_matrix[j, i] = 0.5
                        Pi_matrix[i, j] = 0.5

        Pi_matrix[~skeleton] = 0.0
        np.fill_diagonal(Pi_matrix, 0.0)

        self.polarity_matrix_ = Pi_matrix
        df_Pi = pd.DataFrame(Pi_matrix, index=gene_names, columns=gene_names)

        print("  [+] Polarity confidence matrix Pi assembled (rows=target, cols=source).")
        print("="*60 + "\n")
        return df_Pi

    def _compute_hsic(self, x: np.ndarray, y: np.ndarray) -> float:
        x_col = x.reshape(-1, 1)
        y_col = y.reshape(-1, 1)
        n = x.shape[0]

        # Full-sample HSIC — no subsampling
        idx = np.arange(n)

        dist_x = pdist(x_col[idx], 'sqeuclidean')
        sigma_sq_x = np.median(dist_x) if np.median(dist_x) > 1e-8 else 1.0

        dist_y = pdist(y_col[idx], 'sqeuclidean')
        sigma_sq_y = np.median(dist_y) if np.median(dist_y) > 1e-8 else 1.0

        K_x = rbf_kernel(x_col, gamma=1.0 / (2 * sigma_sq_x))
        K_y = rbf_kernel(y_col, gamma=1.0 / (2 * sigma_sq_y))

        H = np.eye(n) - np.ones((n, n)) / n
        K_xc = K_x @ H
        K_yc = K_y @ H
        hsic_val = np.sum(K_xc * K_yc.T) / ((n - 1) ** 2)

        return hsic_val

    def _normalize_variance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_col = x.reshape(-1, 1)

        # Nonlinear trend removal when poly_degree > 1
        if self.poly_degree > 1:
            poly_x = PolynomialFeatures(degree=self.poly_degree, include_bias=False)
            x_poly = poly_x.fit_transform(x_col)
            model = RidgeCV(cv=3)
            model.fit(x_poly, y)
            residuals = y - model.predict(x_poly)
        else:
            ols = LinearRegression().fit(x_col, y)
            residuals = y - ols.predict(x_col)

        sq_residuals = residuals ** 2

        knn = KNeighborsRegressor(n_neighbors=self.knn_k)
        knn.fit(x_col, sq_residuals)
        knn_var = knn.predict(x_col)

        jitter = np.random.normal(0, 1e-12, size=x.shape)
        x_jittered = x + jitter
        sort_idx = np.argsort(x_jittered)

        x_sorted = x_jittered[sort_idx]
        y_sorted = knn_var[sort_idx]

        x_unique, unique_idx = np.unique(np.round(x_sorted, decimals=10), return_index=True)
        y_unique = y_sorted[unique_idx]

        if len(x_unique) > 10:
            spline = UnivariateSpline(x_unique, y_unique, s=len(x_unique)*0.01)
            var_pred = spline(x)
        else:
            var_pred = knn_var

        var_pred = np.clip(var_pred, 1e-8, None)
        r_normalized = residuals / np.sqrt(var_pred)
        return r_normalized
