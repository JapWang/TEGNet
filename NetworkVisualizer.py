import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import rcParams
import seaborn as sns

class NetworkVisualizer:
    """
    TEG-Net network inference visualization module.
    Covers: heteroskedasticity evidence, HSIC causal rupture quantification,
    macro-scale stationary topology, and micro-scale dose-response kinetics.
    """

    COLOR_POS = "#E64B35"      # Activation (red)
    COLOR_NEG = "#4A7FF0"      # Inhibition (blue)
    COLOR_NODE = "#F3E3C3"     # Node fill
    COLOR_HSIC_FWD = "#00A087" # Forward HSIC (low independence / high confidence)
    COLOR_HSIC_BWD = "#3C5488" # Backward HSIC (high dependence / low confidence)

    @staticmethod
    def setup_matplotlib():
        """Initialize global font settings."""
        rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
        rcParams["axes.unicode_minus"] = False

    @classmethod
    def plot_heteroskedastic_asymmetry(
        cls, df_X: pd.DataFrame, source: str, target: str, save_path: str | None = None
    ):
        """Panel A: Heteroskedastic fan scatter plot — raw transcriptional noise asymmetry."""
        if source not in df_X.columns or target not in df_X.columns: return

        x, y = df_X[source].values, df_X[target].values
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
        scatter_kwargs = dict(facecolors='none', edgecolors='#8491B4', s=30, alpha=0.6, linewidths=1.2)

        # Forward: Source -> Target
        ax1.scatter(x, y, **scatter_kwargs)
        m1, b1 = np.polyfit(x, y, 1)
        ax1.plot(x, m1 * x + b1, color=cls.COLOR_POS, lw=2.5, linestyle='--')
        ax1.set_xlabel(f"Source: {source} (Cause)", fontsize=11, fontweight='bold')
        ax1.set_ylabel(f"Target: {target} (Effect)", fontsize=11, fontweight='bold')
        ax1.set_title(f"Forward: {source} $\\rightarrow$ {target}", fontsize=13, fontweight='bold')

        # Backward: Target -> Source
        ax2.scatter(y, x, **scatter_kwargs)
        m2, b2 = np.polyfit(y, x, 1)
        ax2.plot(y, m2 * y + b2, color=cls.COLOR_NEG, lw=2.5, linestyle='--')
        ax2.set_xlabel(f"Target: {target} (False Cause)", fontsize=11, fontweight='bold')
        ax2.set_ylabel(f"Source: {source} (False Effect)", fontsize=11, fontweight='bold')
        ax2.set_title(f"Backward: {target} $\\rightarrow$ {source}", fontsize=13, fontweight='bold')

        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        fig.suptitle("Transcriptional Heteroskedasticity (Raw Standardized Data)", fontsize=15, fontweight='bold', y=1.05)
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=300, facecolor="#FFFFFF")
        plt.close(fig)

    @classmethod
    def plot_hsic_asymmetry_evidence(
        cls, hsic_matrix: np.ndarray, node_names: list[str], top_k: int = 15, save_path: str | None = None
    ):
        """
        Panel B: HSIC independence asymmetry tornado chart.
        Displays bidirectional HSIC scores — the direction with lower HSIC
        (more independent) is the causal source.
        """
        n_features = hsic_matrix.shape[0]
        edges = []

        for i in range(n_features):
            for j in range(i + 1, n_features):
                if hsic_matrix[i, j] > 0 and hsic_matrix[j, i] > 0:
                    if hsic_matrix[i, j] < hsic_matrix[j, i]:
                        src, tgt = i, j
                        hsic_fwd, hsic_bwd = hsic_matrix[i, j], hsic_matrix[j, i]
                    else:
                        src, tgt = j, i
                        hsic_fwd, hsic_bwd = hsic_matrix[j, i], hsic_matrix[i, j]

                    diff = hsic_bwd - hsic_fwd
                    edges.append((diff, src, tgt, hsic_fwd, hsic_bwd))

        if not edges: return

        edges.sort(key=lambda x: x[0], reverse=True)
        top_edges = edges[:top_k]

        labels = [f"{node_names[src]} $\\rightarrow$ {node_names[tgt]}" for _, src, tgt, _, _ in top_edges]
        fwd_vals = [fwd for _, _, _, fwd, _ in top_edges]
        bwd_vals = [bwd for _, _, _, _, bwd in top_edges]

        fig, ax = plt.subplots(figsize=(8, max(5, top_k * 0.4)))

        y_pos = np.arange(len(labels))
        ax.barh(y_pos, -np.array(bwd_vals), align='center', color=cls.COLOR_HSIC_BWD, label='Backward (False Cause, High HSIC)')
        ax.barh(y_pos, np.array(fwd_vals), align='center', color=cls.COLOR_HSIC_FWD, label='Forward (True Cause, Low HSIC)')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontweight='bold')
        ax.invert_yaxis()
        ax.set_xlabel("HSIC Independence Score\n(Smaller is more independent)", fontsize=11, fontweight='bold')
        ax.set_title(f"Top {top_k} Causal Asymmetry Evidence (HSIC Rupture)", fontsize=14, fontweight='bold', pad=15)

        locs, _ = plt.xticks()
        plt.xticks(locs, [f"{abs(loc):.3f}" for loc in locs])

        ax.axvline(0, color='black', linewidth=1.2)
        ax.legend(loc='lower right', frameon=False, fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=300, facecolor="#FFFFFF")
        plt.close(fig)

    @classmethod
    def plot_polarity_heatmap(cls, df_Pi: pd.DataFrame, save_path: str | None = None):
        """
        Panel C: Polarity confidence matrix Pi heatmap.
        Rows = target genes, Columns = source genes.
        """
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(df_Pi, cmap="YlOrRd", annot=False, cbar_kws={'label': 'Causal Confidence (Soft-Mask)'},
                    linewidths=0.5, ax=ax, vmin=0, vmax=1.0)

        ax.set_title(r"Causal Polarity Matrix $\mathbf{\Pi}$ (Row=Target, Col=Source)", fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel("Target Gene (Effect)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Source Gene (Cause)", fontsize=12, fontweight='bold')

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=300, facecolor="#FFFFFF")
        plt.close(fig)

    @classmethod
    def plot_regulatory_response_curve(
        cls, df_X: pd.DataFrame, J_tensor: np.ndarray, source_idx: int, target_idx: int,
        node_names: list[str], save_path: str | None = None
    ):
        """Panel D: Nonlinear dose-response kinetics curve from occupancy-gated flux."""
        src_name, tgt_name = node_names[source_idx], node_names[target_idx]
        x_concentration = df_X.iloc[:, source_idx].values
        y_flux = J_tensor[:, target_idx, source_idx]

        if np.all(y_flux == 0): return

        fig, ax = plt.subplots(figsize=(6, 4.5))
        is_positive = np.mean(y_flux) > 0
        color = cls.COLOR_POS if is_positive else cls.COLOR_NEG

        ax.scatter(x_concentration, y_flux, facecolors='none', edgecolors='#A0B1BA', s=25, alpha=0.5, zorder=2)
        sort_idx = np.argsort(x_concentration)
        ax.plot(x_concentration[sort_idx], y_flux[sort_idx], color=color, lw=3, zorder=3, label="Occupancy Kinetics")

        ax.axhline(0, color='gray', linestyle='--', linewidth=1.0, zorder=1)
        ax.set_title(f"Micro-Regulatory Kinetics\n{src_name} $\\rightarrow$ {tgt_name}", fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel(f"Source Concentration [{src_name}]", fontsize=11, fontweight='bold')
        ax.set_ylabel(r"Specific Regulatory Flux ($\mathcal{J}$)", fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)

    @classmethod
    def plot_macro_network(cls, df_W_global: pd.DataFrame, edge_threshold: float = 1e-4, save_path: str | None = None):
        """Render TEG-Net global macroscopic physical network (W_global)."""
        cls._render_static_graph(df_W_global.values, df_W_global.columns.tolist(), "Global Macroscopic Physical Laws ($W_{global}$)", edge_threshold, save_path, 1200)

    @classmethod
    def plot_micro_network_snapshot(cls, J_tensor: np.ndarray, cell_idx: int, node_names: list[str], edge_threshold: float = 1e-4, save_path: str | None = None):
        """Render single-cell transient network slice from micro-tensor."""
        cls._render_static_graph(J_tensor[cell_idx], node_names, f"Microscopic Specific Network (Cell #{cell_idx})", edge_threshold, save_path, 800)

    @classmethod
    def _render_static_graph(cls, adj_matrix: np.ndarray, node_names: list[str], title: str, edge_threshold: float, save_path: str | None, node_size_scale: int):
        """Low-level signed directed network rendering via NetworkX."""
        n = adj_matrix.shape[0]
        G = nx.DiGraph()
        node_colors, node_sizes, labels_dict = [], [], {}

        in_degrees = np.sum(np.abs(adj_matrix), axis=1)
        d_min, d_max = np.min(in_degrees), np.max(in_degrees)

        for i in range(n):
            G.add_node(i)
            labels_dict[i] = node_names[i]
            node_colors.append(cls.COLOR_NODE)
            sz = node_size_scale * 0.4 + ((in_degrees[i] - d_min) / (d_max - d_min + 1e-9)) * node_size_scale
            node_sizes.append(sz)

        edges_list, edge_colors, edge_widths, weights = [], [], [], []
        for tgt in range(n):
            for src in range(n):
                if tgt != src and abs(adj_matrix[tgt, src]) >= edge_threshold:
                    G.add_edge(src, tgt, weight=adj_matrix[tgt, src])
                    edges_list.append((src, tgt))
                    weights.append(adj_matrix[tgt, src])

        if not edges_list: return

        max_w = max(abs(w) for w in weights)
        for w in weights:
            edge_colors.append(cls.COLOR_POS if w > 0 else cls.COLOR_NEG)
            edge_widths.append(1.0 + 4.0 * (abs(w) / max_w))

        pos = nx.circular_layout(G)
        fig, ax = plt.subplots(figsize=(8, 8))

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, edgecolors="#2B2B2B", linewidths=1.5)
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges_list, edge_color=edge_colors, width=edge_widths, connectionstyle="arc3,rad=0.15", arrowstyle="-|>", arrowsize=18, node_size=node_sizes)

        # Offset labels with white background bbox to prevent occlusion
        pos_labels = {k: v * 1.12 if np.linalg.norm(v) > 0 else v for k, v in pos.items()}
        bbox_props = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7)
        nx.draw_networkx_labels(G, pos_labels, labels_dict, font_family="sans-serif", font_weight="bold", font_color="#2B2B2B", ax=ax, font_size=11, bbox=bbox_props)

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.margins(0.25)
        ax.axis("off")
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=300, facecolor="#FFFFFF")
        plt.close(fig)
