import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from scipy.stats import spearmanr

class TEGNetEvaluator:
    """
    TEG-Net official evaluation suite (SCI Q1 standard).
    Features:
    1. Evaluates both global network topology (W_global) and micro-gating tensor (J_tensor).
    2. Uses Spearman rank correlation for fairness on nonlinear gating flux.
    3. Enforces diagonal masking — self-regulatory terms excluded from TP/FP counts.
    4. Top-K density alignment for fair comparison with ground truth.
    """

    def __init__(self, truth_path: str, output_dir: str = "TEGNet_Evaluation_Output", true_tensor_path: str = None):
        self.truth = pd.read_csv(truth_path, index_col=0).astype(float)
        self.all_genes = list(self.truth.columns)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Extract ground truth directed edge count K for Top-K alignment
        offdiag = ~np.eye(len(self.all_genes), dtype=bool)
        self.true_edge_count = int(np.sum((self.truth.values != 0) & offdiag))
        print(f"==================================================")
        print(f" [Ground Truth] True directed edge count (K): {self.true_edge_count}")
        print(f"==================================================")

        self.true_tensor = None
        if true_tensor_path and os.path.exists(true_tensor_path):
            self.true_tensor = np.load(true_tensor_path)
            print(f" [Ground Truth] Loaded true micro-dynamic tensor, shape: {self.true_tensor.shape}")

    # ==================== Utilities: alignment & Top-K ====================
    @staticmethod
    def _load_adj(csv_path: str) -> pd.DataFrame:
        adj = pd.read_csv(csv_path, index_col=0).astype(float)
        if adj.isnull().values.any():
            adj = adj.fillna(0.0)
        return adj

    @staticmethod
    def _load_tensor(npy_path: str) -> np.ndarray:
        return np.load(npy_path)

    @staticmethod
    def align_matrix_to_truth(pred_adj: pd.DataFrame, truth_genes: list) -> pd.DataFrame:
        common = [g for g in truth_genes if g in pred_adj.columns]
        aligned = pd.DataFrame(0.0, index=truth_genes, columns=truth_genes)
        aligned.loc[common, common] = pred_adj.loc[common, common]
        return aligned

    @staticmethod
    def align_tensor_to_truth(tensor: np.ndarray, tensor_gene_order: list, truth_genes: list) -> np.ndarray:
        common = [g for g in truth_genes if g in tensor_gene_order]
        idx_tensor = [tensor_gene_order.index(g) for g in common]
        idx_truth = [truth_genes.index(g) for g in common]
        n_cells = tensor.shape[0]
        M = len(truth_genes)
        aligned = np.zeros((n_cells, M, M))
        aligned[np.ix_(range(n_cells), idx_truth, idx_truth)] = tensor[np.ix_(range(n_cells), idx_tensor, idx_tensor)]
        return aligned

    @staticmethod
    def keep_top_k_edges(adj_df: pd.DataFrame, k: int) -> pd.DataFrame:
        mat = adj_df.values.copy()
        np.fill_diagonal(mat, 0.0)
        abs_mat = np.abs(mat)

        flat_weights = abs_mat.ravel()
        sorted_weights = np.sort(flat_weights)

        if k > len(sorted_weights): k = len(sorted_weights)
        threshold = sorted_weights[-k] if k > 0 else np.inf
        if threshold <= 0: threshold = 1e-12

        sparse_mat = np.where(abs_mat >= threshold, mat, 0.0)
        return pd.DataFrame(sparse_mat, index=adj_df.index, columns=adj_df.columns)

    # ==================== Core topology metrics ====================
    def evaluate_adjacency(self, pred_adj_df: pd.DataFrame, truth_genes=None):
        if truth_genes is None: truth_genes = self.all_genes
        aligned = self.align_matrix_to_truth(pred_adj_df, truth_genes)
        W = aligned.values.copy()
        T = self.truth.loc[truth_genes, truth_genes].values.copy()

        # Zero out diagonal (self-regulatory terms)
        np.fill_diagonal(W, 0.0)
        np.fill_diagonal(T, 0.0)

        offdiag = ~np.eye(len(truth_genes), dtype=bool)

        y_true = (T != 0).astype(int).ravel()
        y_score = np.abs(W).ravel()
        mask = offdiag.ravel()

        y_true_off = y_true[mask]
        y_score_off = y_score[mask]

        auroc = roc_auc_score(y_true_off, y_score_off) if len(np.unique(y_true_off)) > 1 else np.nan
        aupr = average_precision_score(y_true_off, y_score_off) if len(np.unique(y_true_off)) > 1 else np.nan

        threshold = 1e-5
        pred_edges = (np.abs(W) > threshold) & offdiag
        true_edges = (T != 0) & offdiag

        TP = int(np.sum(pred_edges & true_edges))
        FP = int(np.sum(pred_edges & ~true_edges & offdiag))
        FN = int(np.sum(~pred_edges & true_edges & offdiag))
        TN = int(np.sum(~pred_edges & ~true_edges & offdiag))

        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        Specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        Precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        F1 = 2 * (Precision * TPR) / (Precision + TPR) if (Precision + TPR) > 0 else 0.0

        denom = np.sqrt(float((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
        MCC = ((TP * TN) - (FP * FN)) / denom if denom > 0 else 0.0

        common = pred_edges & true_edges
        correct_sign = common & (np.sign(W) == np.sign(T))
        wrong_sign = common & (np.sign(W) != np.sign(T))

        signed_TP = int(np.sum(correct_sign))
        sign_error = int(np.sum(wrong_sign))
        sign_acc = np.mean(np.sign(W[common]) == np.sign(T[common])) if np.any(common) else np.nan

        real_flat = true_edges.ravel()
        spear_r = spearmanr(W.ravel()[real_flat], T.ravel()[real_flat])[0] if np.sum(real_flat) > 3 else np.nan

        return {
            "AUROC": auroc, "AUPRC": aupr, "MCC": MCC, "F1_Score": F1,
            "Precision": Precision, "Recall (TPR)": TPR, "Specificity (TNR)": Specificity,
            "FPR": FPR, "Sign_Accuracy": sign_acc,
            "Signed_TP_Edges": signed_TP, "Sign_Error_Edges": sign_error,
            "TP": TP, "FP": FP, "TN": TN, "FN": FN,
            "n_pred_edges": int(np.sum(pred_edges)), "Spearman_R": spear_r,
        }, W, T, offdiag, truth_genes

    # ==================== Visualization ====================
    def plot_roc_pr(self, W, T, offdiag, metrics, suffix="global"):
        y_true = (T != 0).astype(int)[offdiag]
        y_score = np.abs(W)[offdiag]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            axes[0].plot(fpr, tpr, label=f"AUROC = {metrics['AUROC']:.3f}", color='#3C5488', lw=2)
        axes[0].plot([0,1],[0,1], color='gray', linestyle='--', alpha=0.6)
        axes[0].set_title(f"ROC Curve ({suffix})", fontweight='bold')
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].legend(loc="lower right")

        if len(np.unique(y_true)) > 1:
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            axes[1].plot(recall, precision, label=f"AUPRC = {metrics['AUPRC']:.3f}", color='#E64B35', lw=2)
        baseline_pr = np.mean(y_true)
        axes[1].plot([0,1],[baseline_pr, baseline_pr], color='gray', linestyle='--', alpha=0.6)
        axes[1].set_title(f"PR Curve ({suffix})", fontweight='bold')
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].legend(loc="upper right")

        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.tight_layout()
        path = os.path.join(self.output_dir, f"roc_pr_{suffix}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

    def select_critical_edges(self, gene_list, top_k=5):
        T = self.truth.loc[gene_list, gene_list].values
        offdiag = ~np.eye(len(gene_list), dtype=bool)
        rows, cols = np.where(offdiag & (T != 0))
        weights = T[rows, cols]
        if len(weights) == 0: return []
        order = np.argsort(-np.abs(weights))[:min(top_k, len(weights))]
        return [(cols[i], rows[i], weights[i]) for i in order]

    def plot_edge_trajectories(self, tensor, gene_list, suffix="global", true_tensor=None):
        critical = self.select_critical_edges(gene_list)
        if not critical: return
        n = len(critical)
        fig, axes = plt.subplots(n, 1, figsize=(8, 2.5*n), sharex=True)
        if n == 1: axes = [axes]
        t_axis = np.arange(tensor.shape[0])
        for ax, (src, tgt, w) in zip(axes, critical):
            ax.plot(t_axis, tensor[:, tgt, src], color='#3C5488', alpha=0.9, linewidth=1.5, label='TEG-Net Inferred')
            if true_tensor is not None:
                ax.plot(t_axis, true_tensor[:, tgt, src], color='#E64B35', linestyle='--', linewidth=1.5, label='Ground Truth (ODE)')
                ax.legend(loc='best', frameon=False)
            ax.axhline(0, color='gray', linestyle=':', alpha=0.6)

            ax.set_ylabel(f"Occupancy-Gated Flux\n{gene_list[tgt]} $\\leftarrow$ {gene_list[src]}")
            ax.set_title(f"{gene_list[src]} $\\rightarrow$ {gene_list[tgt]} (True Weight: {w:+.1f})", fontweight='bold', fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        axes[-1].set_xlabel("Cell Sample Index (Observation Order)")
        plt.tight_layout()
        path = os.path.join(self.output_dir, f"critical_edge_trajectories_{suffix}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_individual_similarity(self, tensor, global_adj_df, gene_list, suffix="global"):
        W_global = self.align_matrix_to_truth(global_adj_df, gene_list).values
        offdiag = ~np.eye(len(gene_list), dtype=bool)
        global_vec = W_global[offdiag]
        similarities = []
        for k in range(tensor.shape[0]):
            indi_vec = tensor[k][offdiag]
            norm = np.linalg.norm(global_vec) * np.linalg.norm(indi_vec)
            if norm < 1e-12: similarities.append(np.nan)
            else: similarities.append(np.dot(global_vec, indi_vec) / norm)
        sim_arr = np.array(similarities)
        valid = sim_arr[~np.isnan(sim_arr)]
        if len(valid) == 0: return valid

        fig, ax = plt.subplots(figsize=(6,4))
        ax.hist(valid, bins=25, color='#4DBBD5', edgecolor='white', alpha=0.9)
        ax.axvline(np.median(valid), color='#E64B35', linestyle='--', linewidth=2, label=f"Median = {np.median(valid):.3f}")
        ax.set_xlabel("Cosine Similarity (Individual vs Population)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Dynamic Heterogeneity Distribution", fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False)
        plt.tight_layout()
        path = os.path.join(self.output_dir, f"individual_similarity_{suffix}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return valid

    # ==================== Tensor evaluation ====================
    def _evaluate_micro_tensor(self, pred_tensor, true_tensor, gene_list):
        """Compare predicted vs ground truth regulatory tensors."""
        T = self.truth.loc[gene_list, gene_list].values
        offdiag = ~np.eye(len(gene_list), dtype=bool)
        rows, cols = np.where(offdiag & (T != 0))
        rmse_list, mae_list, spearman_r_list, r2_list = [], [], [], []

        for r, c in zip(rows, cols):
            true_seq = true_tensor[:, r, c]
            pred_seq = pred_tensor[:, r, c]
            rmse = np.sqrt(np.mean((pred_seq - true_seq)**2))
            mae = np.mean(np.abs(pred_seq - true_seq))

            if np.std(true_seq) > 1e-8 and np.std(pred_seq) > 1e-8:
                r_val = spearmanr(true_seq, pred_seq)[0]
            else:
                r_val = np.nan

            ss_tot = np.sum((true_seq - np.mean(true_seq))**2)
            ss_res = np.sum((true_seq - pred_seq)**2)
            r2_val = 1 - (ss_res / ss_tot) if ss_tot > 1e-8 else np.nan

            rmse_list.append(rmse); mae_list.append(mae)
            spearman_r_list.append(r_val); r2_list.append(r2_val)

        return {
            "MicroTensor_Mean_RMSE": np.mean(rmse_list) if rmse_list else np.nan,
            "MicroTensor_Mean_MAE": np.mean(mae_list) if mae_list else np.nan,
            "MicroTensor_Mean_Spearman_r": np.nanmean(spearman_r_list) if spearman_r_list else np.nan,
            "MicroTensor_Mean_R2": np.nanmean(r2_list) if r2_list else np.nan
        }

    # ==================== Main evaluation workflows ====================
    def evaluate_teg_net(self, global_adj_path: str, tensor_path: str):
        """Full TEG-Net evaluation: sparse network + density-aligned + micro-tensor."""
        if not os.path.exists(global_adj_path): raise FileNotFoundError(f"Network matrix not found: {global_adj_path}")
        if not os.path.exists(tensor_path): raise FileNotFoundError(f"Transient tensor not found: {tensor_path}")

        adj = self._load_adj(global_adj_path)
        tensor_raw = self._load_tensor(tensor_path)
        gene_order = list(adj.columns)
        common_genes = [g for g in self.all_genes if g in gene_order]

        adj = adj.loc[common_genes, common_genes]
        tensor = self.align_tensor_to_truth(tensor_raw, gene_order, common_genes)

        print("\n>>> [1/3] Evaluating TEG-Net raw sparse network skeleton...")
        metrics_raw, W, T, offdiag, _ = self.evaluate_adjacency(adj, common_genes)
        self.plot_roc_pr(W, T, offdiag, metrics_raw, "teg_raw")

        print(f">>> [2/3] Evaluating TEG-Net Top-{self.true_edge_count} density-aligned network...")
        adj_sparse = self.keep_top_k_edges(adj, k=self.true_edge_count)
        metrics_topk, W_sparse, T_sparse, offdiag_sparse, _ = self.evaluate_adjacency(adj_sparse, common_genes)
        self.plot_roc_pr(W_sparse, T_sparse, offdiag_sparse, metrics_topk, "teg_topk")

        print(">>> [3/3] Deep micro-tensor occupancy kinetics validation...")
        true_jac = None
        if self.true_tensor is not None:
            idx = [self.all_genes.index(g) for g in common_genes]
            true_jac = self.true_tensor[np.ix_(range(tensor.shape[0]), idx, idx)]
            jac_metrics = self._evaluate_micro_tensor(tensor, true_jac, common_genes)
            metrics_raw.update(jac_metrics)
            metrics_topk.update(jac_metrics)

        self.plot_edge_trajectories(tensor, common_genes, "teg_net", true_jac)
        sim_arr = self.plot_individual_similarity(tensor, adj, common_genes, "teg_net")
        metrics_topk["Individual_Sim_Mean"] = float(np.mean(sim_arr)) if len(sim_arr)>0 else np.nan

        combined_metrics = {}
        for k, v in metrics_raw.items(): combined_metrics[f"[Raw] {k}"] = v
        for k, v in metrics_topk.items(): combined_metrics[f"[Top-{self.true_edge_count}] {k}"] = v

        summary_text = (
            f"--- TEG-Net Core Metrics (Truth K = {self.true_edge_count}) ---\n"
            f"  > AUROC    | Raw: {metrics_raw['AUROC']:.4f} \t-> Top-K: {metrics_topk['AUROC']:.4f}\n"
            f"  > AUPRC    | Raw: {metrics_raw['AUPRC']:.4f} \t-> Top-K: {metrics_topk['AUPRC']:.4f}\n"
            f"  > MCC      | Raw: {metrics_raw['MCC']:.4f} \t-> Top-K: {metrics_topk['MCC']:.4f}\n"
            f"  > Sign Acc | Raw: {metrics_raw['Sign_Accuracy']:.4f} \t-> Top-K: {metrics_topk['Sign_Accuracy']:.4f}\n"
        )
        if self.true_tensor is not None:
            summary_text += f"  > MicroTensor Spearman r: {metrics_topk.get('MicroTensor_Mean_Spearman_r', np.nan):.4f}\n"

        print(f"\n{summary_text}")
        self._write_report(combined_metrics, "TEG-Net", custom_summary=summary_text)
        return metrics_topk

    def evaluate_baseline(self, global_adj_path: str, method_name: str = "Baseline"):
        """
        Baseline evaluation for static-only methods (GENIE3, GRNBoost2, etc.).
        Skips tensor analysis; performs core topology Top-K evaluation only.
        """
        if not os.path.exists(global_adj_path):
            raise FileNotFoundError(f"Network matrix not found: {global_adj_path}")

        adj = self._load_adj(global_adj_path)
        gene_order = list(adj.columns)
        common_genes = [g for g in self.all_genes if g in gene_order]

        adj = adj.loc[common_genes, common_genes]

        print(f"\n>>> [1/2] Evaluating {method_name} raw inferred network...")
        metrics_raw, W, T, offdiag, _ = self.evaluate_adjacency(adj, common_genes)
        self.plot_roc_pr(W, T, offdiag, metrics_raw, f"{method_name}_raw")

        print(f">>> [2/2] Evaluating {method_name} Top-{self.true_edge_count} density-aligned network...")
        adj_sparse = self.keep_top_k_edges(adj, k=self.true_edge_count)
        metrics_topk, W_sparse, T_sparse, offdiag_sparse, _ = self.evaluate_adjacency(adj_sparse, common_genes)
        self.plot_roc_pr(W_sparse, T_sparse, offdiag_sparse, metrics_topk, f"{method_name}_topk")

        combined_metrics = {}
        for k, v in metrics_raw.items(): combined_metrics[f"[Raw] {k}"] = v
        for k, v in metrics_topk.items(): combined_metrics[f"[Top-{self.true_edge_count}] {k}"] = v

        summary_text = (
            f"--- {method_name} Core Metrics (Truth K = {self.true_edge_count}) ---\n"
            f"  > AUROC    | Raw: {metrics_raw['AUROC']:.4f} \t-> Top-K: {metrics_topk['AUROC']:.4f}\n"
            f"  > AUPRC    | Raw: {metrics_raw['AUPRC']:.4f} \t-> Top-K: {metrics_topk['AUPRC']:.4f}\n"
            f"  > MCC      | Raw: {metrics_raw['MCC']:.4f} \t-> Top-K: {metrics_topk['MCC']:.4f}\n"
            f"  > Sign Acc | Raw: {metrics_raw['Sign_Accuracy']:.4f} \t-> Top-K: {metrics_topk['Sign_Accuracy']:.4f}\n"
        )

        print(f"\n{summary_text}")
        self._write_report(combined_metrics, method_name, custom_summary=summary_text)
        return metrics_topk

    # ==================== Report output ====================
    def _write_report(self, metrics, method_name, custom_summary=""):
        report_path = os.path.join(self.output_dir, f"{method_name}_evaluation_report.txt")
        with open(report_path, "w", encoding="utf-8-sig") as f:
            f.write(f"[{method_name}] Algorithm Topology and Specificity Flux Evaluation Report\n")
            f.write("=" * 65 + "\n")
            f.write(f"Evaluation baseline: Top-K density matching with K = {self.true_edge_count} (true directed edges)\n")
            f.write("=" * 65 + "\n\n")

            if custom_summary:
                f.write("[Core Metrics Summary]\n")
                f.write(custom_summary)
                f.write("=" * 65 + "\n\n")

            f.write("[Detailed Statistical Metrics]\n")
            for k, v in metrics.items():
                if isinstance(v, float): f.write(f"  {k}: {v:.4f}\n")
                else: f.write(f"  {k}: {v}\n")

            f.write(f"\n[Note]: Metrics labeled 'Top-{self.true_edge_count}' retain only the top {self.true_edge_count} off-diagonal edges by absolute weight and zero out the rest. This is the international standard for fair comparison against dense network baselines.\n")

        print(f"[OK] {method_name} evaluation report saved to: {self.output_dir}")


# ==================== Quick-start guide ====================
if __name__ == "__main__":

    # 1. Specify ground truth file location
    TRUTH_CSV = os.path.join("benchmark_data_HSC", "clean_1", "gold_standard_network.csv")

    # Set to None if no ground truth tensor available
    TRUE_TENSOR_NPY = None

    # 2. Specify TEG-Net output locations
    OUTPUT_DIR_TEG = os.path.join("results_HSC", "clean_1", "TEGNet_Evaluation_Results")
    TEG_W_CSV = os.path.join("results_HSC", "clean_1", "TEGNet_results_improved", "TEGNet_Global_Macro_Network_W.csv")
    TEG_J_NPY = os.path.join("results_HSC", "clean_1", "TEGNet_results_improved", "TEGNet_Micro_Tensor_J.npy")

    if os.path.exists(TEG_W_CSV) and os.path.exists(TEG_J_NPY):
        evaluator_teg = TEGNetEvaluator(truth_path=TRUTH_CSV, output_dir=OUTPUT_DIR_TEG, true_tensor_path=TRUE_TENSOR_NPY)
        evaluator_teg.evaluate_teg_net(global_adj_path=TEG_W_CSV, tensor_path=TEG_J_NPY)
    else:
        print("Please run TEG-Net first to generate W_global.csv and J_tensor.npy.")
