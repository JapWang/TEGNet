import os
import time
import numpy as np
import pandas as pd

from DataPreprocessor import DataPreprocessor
from HeteroskedasticCausalInferer import HeteroskedasticCausalInferer
from STRREngine import STRREngine
from OccupancyGating import OccupancyGating
from NetworkVisualizer import NetworkVisualizer

def run_tegnet_pipeline(
    input_csv: str,
    output_dir: str = "TEGNet_Output",
    n_features: int = None,
    knn_k: int = 15,
    ebic_gamma: float = 0.5,
    n_hill: float = 2.0,
    poly_degree: int = 1
):
    """
    TEG-Net (Transcriptional-noise Empowered Causal Network) central dispatch engine.
    Executes fully automated end-to-end single-cell causal network inference and visualization.
    """
    start_time = time.time()
    print("\n" + "="*70)
    print("      TEG-Net (Transcriptional-noise Empowered Causal Network)      ")
    print("      - End-to-End Inference Pipeline -     ")
    print("="*70)

    os.makedirs(output_dir, exist_ok=True)
    NetworkVisualizer.setup_matplotlib()

    try:
        # Module 1: Data Preprocessing
        print(f"\n[Module 1] Loading static expression cross-section: {input_csv}")
        df_X = DataPreprocessor.load_data(input_csv, n_features=n_features, standardize=True)
        gene_names = df_X.columns.tolist()
        print(f"  -> Data shape: {df_X.shape[0]} cells x {df_X.shape[1]} genes. Standardization complete.")

        # Module 2: Heteroskedastic Causal Polarity Inference
        print(f"\n[Module 2] Starting single-cell transcriptional burst heteroskedastic polarity analysis (poly_degree={poly_degree})...")
        inferer = HeteroskedasticCausalInferer(knn_k=knn_k, ebic_gamma=ebic_gamma, poly_degree=poly_degree)
        df_Pi = inferer.fit_transform(df_X)

        pi_path = os.path.join(output_dir, "TEGNet_Polarity_Matrix_Pi.csv")
        df_Pi.to_csv(pi_path)
        print(f"  -> [Saved] Polarity confidence matrix: {pi_path}")

        # Module 3: STRR — Stationary Physical Network Recovery
        print("\n[Module 3] Starting soft-mask sequential threshold ridge regression (STRR)...")
        strr = STRREngine(n_threshold_paths=50)
        df_W_global = strr.fit_transform(df_X, df_Pi)

        w_path = os.path.join(output_dir, "TEGNet_Global_Macro_Network_W.csv")
        df_W_global.to_csv(w_path)
        print(f"  -> [Saved] Global macro physical network: {w_path}")

        # Module 4: Occupancy Gating — Micro-state Tensor Assembly
        print("\n[Module 4] Assembling 3D micro-evolution tensor (occupancy gating dynamics)...")
        gating = OccupancyGating(n_hill=n_hill)
        J_tensor = gating.fit_transform(df_X, df_W_global)

        j_path = os.path.join(output_dir, "TEGNet_Micro_Tensor_J.npy")
        np.save(j_path, J_tensor)
        print(f"  -> [Saved] Cell-specific micro-causal tensor (N*M*M): {j_path}")

        # Module 5: Visualization Suite
        vis_dir = os.path.join(output_dir, "Visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        print("\n[Module 5] Generating publication-quality figures...")

        # 5.1 Polarity confidence heatmap
        NetworkVisualizer.plot_polarity_heatmap(
            df_Pi, save_path=os.path.join(vis_dir, "00_Polarity_Confidence_Matrix_Pi.png")
        )

        # 5.2 HSIC asymmetry evidence tornado chart
        if hasattr(inferer, 'hsic_matrix_') and inferer.hsic_matrix_ is not None:
            NetworkVisualizer.plot_hsic_asymmetry_evidence(
                inferer.hsic_matrix_, gene_names, top_k=15,
                save_path=os.path.join(vis_dir, "01_HSIC_Asymmetry_Evidence.png")
            )

        # 5.3 Global macro network topology
        NetworkVisualizer.plot_macro_network(
            df_W_global, save_path=os.path.join(vis_dir, "02_Global_Macro_Network.png")
        )

        # 5.4 Strongest regulatory edge: heteroskedasticity + dose-response
        w_mat = df_W_global.values
        if np.any(np.abs(w_mat) > 0):
            max_idx = np.unravel_index(np.argmax(np.abs(w_mat)), w_mat.shape)
            tgt_idx, src_idx = max_idx[0], max_idx[1]
            src_name, tgt_name = gene_names[src_idx], gene_names[tgt_idx]

            NetworkVisualizer.plot_heteroskedastic_asymmetry(
                df_X, src_name, tgt_name, save_path=os.path.join(vis_dir, f"03_Heteroskedasticity_{src_name}_{tgt_name}.png")
            )
            NetworkVisualizer.plot_regulatory_response_curve(
                df_X, J_tensor, src_idx, tgt_idx, gene_names, save_path=os.path.join(vis_dir, f"04_DoseResponse_Kinetics_{src_name}_{tgt_name}.png")
            )

        # 5.5 Single-cell micro-network snapshots (first, middle, last)
        n_samples = df_X.shape[0]
        sample_cells = [0, n_samples // 2, n_samples - 1]
        for c_idx in sample_cells:
            NetworkVisualizer.plot_micro_network_snapshot(
                J_tensor, c_idx, gene_names, save_path=os.path.join(vis_dir, f"05_Micro_Network_Cell_Rank{c_idx}.png")
            )

        elapsed_time = time.time() - start_time
        print("\n" + "="*70)
        print(f"  [Success] TEG-Net inference pipeline complete! Elapsed: {elapsed_time:.2f} s.")
        print(f"  Results saved to: {os.path.abspath(output_dir)}")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n[Fatal Error] TEG-Net pipeline interrupted: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys

    # Reproducibility
    np.random.seed(42)

    # 1. Basic I/O configuration
    INPUT_FILE = os.path.join("benchmark_data_HSC", "clean_1", "gold_standard_expression.csv")
    OUTPUT_DIR = os.path.join("results_HSC", "clean_1", "TEGNet_results")

    # 2. Feature count control (0 = use all genes)
    FEATURES_TO_RUN = 0

    # 3. Core hyperparameters
    GAMMA_VAL = 1.5     # EBIC sparsity penalty (higher = sparser network)
    KNN_VAL = 25        # KNN receptive field for local variance smoothing
    HILL_VAL = 2.0      # Hill cooperativity coefficient (2.0 = classical dimerization)
    POLY_VAL = 1        # Polynomial degree for partial residual extraction (1 = linear, recommended)

    # Auto-detect input file
    if not os.path.exists(INPUT_FILE):
        print("\n" + "!"*60)
        print(f" [Fatal Error] Input file not found: {INPUT_FILE}")
        print(" Please verify the data file path.")
        print("!"*60 + "\n")
        sys.exit(1)

    n_feat = FEATURES_TO_RUN if FEATURES_TO_RUN > 0 else None

    run_tegnet_pipeline(
        input_csv=INPUT_FILE,
        output_dir=OUTPUT_DIR,
        n_features=n_feat,
        knn_k=KNN_VAL,
        ebic_gamma=GAMMA_VAL,
        n_hill=HILL_VAL,
        poly_degree=POLY_VAL
    )
