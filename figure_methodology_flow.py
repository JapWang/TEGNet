"""
@Figure_Methodology_Flow — TEG-Net Algorithm Schematic (4-Panel)
=================================================================
Panel A: Shape-Preserving Manifold (dropout matrix → imputed manifold)
Panel B: Heteroskedastic Causal Rupture (fan scatter + directional breaking)
Panel C: Macroscopic Physical Laws (signed network + GIC filter)
Panel D: Cell-Specific Micro-states (macro→micro projection + Hill kinetics)

Output: Final_Figures/Figure_Methodology_Flow.pdf + .png
"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import networkx as nx

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 8.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.dpi': 600, 'savefig.bbox': 'tight',
})

OUT_DIR = "Final_Figures"
os.makedirs(OUT_DIR, exist_ok=True)

GOLDEN = 1.618
PANEL_W = 6.4
PANEL_H = PANEL_W / GOLDEN  # ≈ 3.95

C = {
    'red':    '#E64B35', 'blue':   '#4A7FF0', 'green':  '#00A087',
    'purple': '#7E57C2', 'orange': '#F39B7F', 'gray':   '#ADB5BD',
    'dark':   '#2C3E50', 'sub':    '#6C757D', 'bg':     '#FFFFFF',
    'cream':  '#F8F9FA', 'pink':   '#FCE4EC',
}

def draw_panel_border(ax, label):
    """Subtle panel border + label."""
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#DEE2E6')
        spine.set_linewidth(0.6)
    ax.text(0.02, 0.97, label, transform=ax.transAxes, fontsize=14,
            fontweight='bold', va='top', ha='left', color=C['dark'])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


# ============================================================
# PANEL A: Shape-Preserving Manifold Initialization
# ============================================================
def panel_a(ax):
    draw_panel_border(ax, 'A')

    # --- Left: dropout matrix (sparse with blank squares) ---
    mat_size = 8
    np.random.seed(1)
    dropout_mask = np.random.choice([0, 1], size=(mat_size, mat_size), p=[0.5, 0.5])
    values = np.random.uniform(0.3, 1.0, size=(mat_size, mat_size))
    values[dropout_mask == 0] = np.nan

    # Draw left matrix as a grid of squares
    left_x, left_y = 0.08, 0.22
    cell_sz = 0.07
    gap = 0.012
    for i in range(mat_size):
        for j in range(mat_size):
            x = left_x + j * (cell_sz + gap)
            y = left_y + (mat_size - 1 - i) * (cell_sz + gap)
            v = values[i, j]
            if np.isnan(v):
                fc = '#FFFFFF'
                ec = '#DEE2E6'
                ax.add_patch(Rectangle((x, y), cell_sz, cell_sz, fc=fc, ec=ec, lw=0.5, zorder=2))
            else:
                fc = plt.cm.YlOrRd(0.2 + v * 0.6)
                ax.add_patch(Rectangle((x, y), cell_sz, cell_sz, fc=fc, ec='white', lw=0.3, zorder=2))

    # Label
    ax.text(left_x + (cell_sz+gap)*mat_size/2, left_y - 0.12,
            r'$\mathbf{X}_{raw}$ (with Dropout)', fontsize=8, ha='center', color=C['sub'])

    # --- Arrow ---
    arrow_x = left_x + (cell_sz+gap)*mat_size + 0.1
    arrow_y = left_y + (cell_sz+gap)*mat_size/2
    ax.annotate('', xy=(arrow_x + 0.55, arrow_y), xytext=(arrow_x + 0.05, arrow_y),
                arrowprops=dict(arrowstyle='->', color=C['red'], lw=2.5), zorder=4)

    # Formula on arrow
    ax.text(arrow_x + 0.3, arrow_y + 0.18,
            r'$\mathbf{X}_{work} = \mathbf{X} + \mathbf{E}$', fontsize=8, ha='center',
            fontweight='bold', color=C['dark'])
    ax.text(arrow_x + 0.3, arrow_y + 0.02,
            r'$\mathbf{E} \sim \mathcal{N}(0, 10^{-12})$', fontsize=7.5, ha='center', color=C['sub'])

    # KNN note
    ax.text(arrow_x + 0.3, arrow_y - 0.14,
            'KNN Imputation (k=5)\nMin-Max Scaling → [0,1]', fontsize=6.5, ha='center', color=C['gray'])

    # --- Right: smooth imputed matrix (gradient) ---
    right_x = arrow_x + 0.7
    smooth_vals = np.random.uniform(0.2, 1.0, size=(mat_size, mat_size))
    for i in range(mat_size):
        for j in range(mat_size):
            x = right_x + j * (cell_sz + gap)
            y = left_y + (mat_size - 1 - i) * (cell_sz + gap)
            v = smooth_vals[i, j]
            fc = plt.cm.YlOrRd(0.15 + v * 0.7)
            ax.add_patch(Rectangle((x, y), cell_sz, cell_sz, fc=fc, ec='white', lw=0.3, zorder=2))

    ax.text(right_x + (cell_sz+gap)*mat_size/2, left_y - 0.12,
            r'$\mathbf{X}_{work}$ (Probability Manifold)', fontsize=8, ha='center', color=C['sub'])

    # --- Descriptive text ---
    ax.text(0.5, 0.92, 'Non-negative Manifold Mapping & Measure-preserving Perturbation',
            transform=ax.transAxes, fontsize=10, fontweight='bold', ha='center', color=C['dark'])
    ax.text(0.5, 0.06, 'Shape-preserving normalization retains endogenous variance gradient critical for downstream heteroskedasticity detection.',
            transform=ax.transAxes, fontsize=7, ha='center', color=C['sub'], fontstyle='italic')


# ============================================================
# PANEL B: Heteroskedastic Causal Rupture
# ============================================================
def panel_b(ax):
    draw_panel_border(ax, 'B')
    ax.text(0.5, 0.92, 'Directional Symmetry Breaking via Transcriptional Noise',
            transform=ax.transAxes, fontsize=10, fontweight='bold', ha='center', color=C['dark'])

    # --- Left 60%: Fan-shaped scatter plot ---
    ax_scatter = ax.inset_axes([0.04, 0.12, 0.52, 0.70])
    ax_scatter.set_facecolor('white')

    # Generate heteroskedastic data: variance increases with X
    np.random.seed(42)
    n_pts = 300
    x = np.random.uniform(0, 1, n_pts)
    x_sort = np.sort(x)
    noise_scale = 0.02 + 0.18 * x_sort  # variance increases with x
    y = 0.3 * x_sort + 0.2 + np.random.normal(0, noise_scale)

    ax_scatter.scatter(x, y, c='#8491B4', s=8, alpha=0.45, edgecolors='none', zorder=2)
    # Trend line
    ax_scatter.plot(x_sort, 0.3 * x_sort + 0.2, color=C['red'], lw=1.8, zorder=3)
    # Variance envelope
    ax_scatter.fill_between(x_sort, 0.3*x_sort+0.2-1.96*noise_scale,
                            0.3*x_sort+0.2+1.96*noise_scale,
                            alpha=0.10, color=C['red'], zorder=1)

    ax_scatter.set_xlabel('Source Gene Expression (X)', fontsize=7.5, color=C['sub'])
    ax_scatter.set_ylabel('Target Gene Expression (Y)', fontsize=7.5, color=C['sub'])
    ax_scatter.set_title('Transcriptional Heteroskedasticity', fontsize=8, fontweight='bold', color=C['dark'], pad=3)
    ax_scatter.tick_params(labelsize=6)
    ax_scatter.spines['top'].set_visible(False)
    ax_scatter.spines['right'].set_visible(False)

    # Annotation: "variance increases →"
    ax_scatter.annotate('Var(Y|X) increases →', xy=(0.55, 0.85), xytext=(0.15, 0.93),
                        fontsize=6.5, color=C['red'], fontweight='bold', ha='center')

    # --- Right 40%: Causal diagram ---
    # Two nodes A and B
    node_x = 0.72
    node_radius = 0.06

    # Node A (left)
    ax.add_patch(plt.Circle((node_x - 0.12, 0.65), node_radius, fc=C['blue'], ec='white', lw=1.5, zorder=5))
    ax.text(node_x - 0.12, 0.65, 'A', fontsize=8, fontweight='bold', color='white', ha='center', va='center', zorder=6)

    # Node B (right)
    ax.add_patch(plt.Circle((node_x + 0.12, 0.65), node_radius, fc=C['red'], ec='white', lw=1.5, zorder=5))
    ax.text(node_x + 0.12, 0.65, 'B', fontsize=8, fontweight='bold', color='white', ha='center', va='center', zorder=6)

    # A → B solid arrow (true causal)
    ax.annotate('', xy=(node_x + 0.12 - node_radius, 0.65),
                xytext=(node_x - 0.12 + node_radius, 0.65),
                arrowprops=dict(arrowstyle='->', color=C['green'], lw=3.0), zorder=4)
    ax.text(node_x, 0.72, 'TRUE', fontsize=7, ha='center', fontweight='bold', color=C['green'])

    # B → A with red X (rejected)
    ax.annotate('', xy=(node_x - 0.12 + node_radius, 0.58),
                xytext=(node_x + 0.12 - node_radius, 0.58),
                arrowprops=dict(arrowstyle='->', color=C['gray'], lw=1.5, linestyle='dashed'), zorder=4)
    # Red X
    ax.text(node_x, 0.545, '✗', fontsize=13, ha='center', color=C['red'], fontweight='bold', zorder=5)
    ax.text(node_x + 0.04, 0.52, 'REJECTED', fontsize=6, color=C['red'], fontweight='bold')

    # HSIC values
    ax.text(node_x, 0.46, r'$\mathrm{HSIC}_{B \to A} \gg \mathrm{HSIC}_{A \to B}$',
            fontsize=7, ha='center', color=C['dark'], fontweight='bold')

    # --- Polarity formula (bottom) ---
    formula_y = 0.18
    ax.text(0.5, formula_y + 0.10,
            r'$\mathbf{\Pi_{j,k} = \frac{HSIC_{j \to k}}{HSIC_{k \to j} + HSIC_{j \to k}}}$',
            fontsize=10.5, ha='center', fontweight='bold', color=C['red'],
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=C['pink'], edgecolor=C['red'], alpha=0.7, lw=1.2))

    ax.text(0.5, formula_y - 0.02, 'Parameter-free HSIC Ratio Polarity Mapping',
            fontsize=7.5, ha='center', color=C['sub'], transform=ax.transAxes)


# ============================================================
# PANEL C: Macroscopic Physical Laws
# ============================================================
def panel_c(ax):
    draw_panel_border(ax, 'C')
    ax.text(0.5, 0.92, r'Strict $L_0$ Sparsity Filter for Global Stationary Topology',
            transform=ax.transAxes, fontsize=10, fontweight='bold', ha='center', color=C['dark'])

    # --- Network visualization (left 55%) ---
    net_ax = ax.inset_axes([0.02, 0.10, 0.55, 0.72])
    net_ax.axis('off')

    G = nx.DiGraph()
    nodes = ['GATA1', 'GATA2', 'PU.1', 'CEBPA', 'FLI1', 'GFI1']
    pos = nx.circular_layout(G)
    # Custom positions for better layout
    angles = np.linspace(0, 2*np.pi, len(nodes), endpoint=False)
    radius = 1.0
    pos = {n: (radius*np.cos(a), radius*np.sin(a)) for n, a in zip(nodes, angles)}
    G.add_nodes_from(nodes)

    # Edges with signs
    edges = [('GATA1','GATA2',+1), ('PU.1','GATA1',-1), ('CEBPA','GATA1',-1),
             ('CEBPA','PU.1',+1), ('FLI1','GATA2',+1), ('GFI1','PU.1',-1),
             ('GATA2','GFI1',+1), ('CEBPA','FLI1',-1)]
    for s, t, w in edges:
        G.add_edge(s, t, weight=w)

    nx.draw_networkx_nodes(G, pos, ax=net_ax, node_size=500,
                           node_color='#F3E3C3', edgecolors='#2B2B2B', linewidths=1.5)
    nx.draw_networkx_labels(G, pos, ax=net_ax, font_size=7.5, font_weight='bold', font_family='sans-serif')

    # Draw edges with colors
    edge_colors = [C['red'] if G[u][v]['weight']>0 else C['blue'] for u,v in G.edges()]
    edge_widths = [3.0 for _ in G.edges()]
    nx.draw_networkx_edges(G, pos, ax=net_ax, edge_color=edge_colors, width=edge_widths,
                           arrowstyle='-|>', arrowsize=15, connectionstyle='arc3,rad=0.12')

    # Legend
    net_ax.plot([], [], '-', color=C['red'], lw=2, label='Activation (+)')
    net_ax.plot([], [], '-', color=C['blue'], lw=2, label='Inhibition (−)')
    net_ax.legend(loc='lower center', frameon=False, fontsize=6.5, ncol=2, markerscale=0.5,
                  bbox_to_anchor=(0.5, -0.08))

    # --- GIC Formula (right 40%) ---
    formula_x = 0.62
    ax.text(formula_x, 0.75, 'Adaptive GIC Filter:', fontsize=9, fontweight='bold', color=C['dark'],
            transform=ax.transAxes)

    ax.text(formula_x, 0.62,
            r'$\mathbf{\mathrm{GIC} = N\ln\left(\frac{RSS}{N}\right)}$',
            fontsize=9.5, fontweight='bold', color=C['dark'], transform=ax.transAxes)
    ax.text(formula_x, 0.54,
            r'$\mathbf{+\; k\ln(N) \cdot \kappa}$',
            fontsize=9.5, fontweight='bold', color=C['dark'], transform=ax.transAxes)

    ax.text(formula_x, 0.44,
            r'$\kappa = \min(2.0,\; 1.0 + \frac{\ln P}{\ln 100})$',
            fontsize=8, fontweight='bold', color=C['red'], transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=C['pink'], edgecolor=C['red'], alpha=0.6, lw=0.8))

    # Explanatory text
    ax.text(formula_x, 0.33, 'Sequential Threshold\nRidge Regression (STRR)', fontsize=7.5,
            transform=ax.transAxes, color=C['sub'])
    ax.text(formula_x, 0.22, '→ Global signed network\n  with optimal sparsity\n→ >90% sign accuracy', fontsize=7,
            transform=ax.transAxes, color=C['green'], fontweight='bold')

    # Output
    ax.text(0.5, 0.06,
            r'Output: $\mathbf{W}_{global} \in \mathbb{R}^{M \times M}$  (signed directed graph, + activation / − inhibition)',
            fontsize=7.5, ha='center', color=C['sub'], transform=ax.transAxes, fontstyle='italic')


# ============================================================
# PANEL D: Cell-Specific Micro-states
# ============================================================
def panel_d(ax):
    draw_panel_border(ax, 'D')
    ax.text(0.5, 0.92, 'Non-linear Occupancy Kinetics for Micro-state Projection',
            transform=ax.transAxes, fontsize=10, fontweight='bold', ha='center', color=C['dark'])

    # --- Top: Macro network (simplified) ---
    macro_ax = ax.inset_axes([0.15, 0.58, 0.70, 0.30])
    macro_ax.axis('off')
    macro_ax.set_facecolor('white')

    G_macro = nx.DiGraph()
    macro_nodes = ['A', 'B', 'C', 'D']
    macro_pos = {'A': (0, 0), 'B': (1, 0.3), 'C': (0.5, 1), 'D': (1, 0.9)}
    G_macro.add_nodes_from(macro_nodes)
    G_macro.add_edges_from([('A','B'), ('A','C'), ('C','D'), ('D','B')])

    nx.draw_networkx_nodes(G_macro, macro_pos, ax=macro_ax, node_size=350,
                           node_color='#F3E3C3', edgecolors='#2B2B2B', linewidths=1.5)
    nx.draw_networkx_labels(G_macro, macro_pos, ax=macro_ax, font_size=8, font_weight='bold')
    macro_colors = [C['red'] if i%2==0 else C['blue'] for i in range(len(G_macro.edges()))]
    nx.draw_networkx_edges(G_macro, macro_pos, ax=macro_ax, edge_color=macro_colors,
                           width=2.5, arrowstyle='-|>', arrowsize=12,
                           connectionstyle='arc3,rad=0.1')
    macro_ax.set_title(r'$\mathbf{W}_{global}$  (Macroscopic Population Network)',
                       fontsize=8, fontweight='bold', color=C['dark'], pad=2)

    # --- Three dashed arrows down ---
    arrow_y_start = 0.56
    arrow_y_end = 0.48
    for x_pos in [0.25, 0.50, 0.75]:
        ax.annotate('', xy=(x_pos, arrow_y_end), xytext=(x_pos, arrow_y_start),
                    arrowprops=dict(arrowstyle='->', color=C['gray'], lw=1.2,
                                   linestyle='dashed'), transform=ax.transAxes, zorder=3)

    # --- Bottom: Three micro networks with different edge thicknesses ---
    cell_labels = ['Cell 1', 'Cell 2', 'Cell 3']
    np.random.seed(123)
    for ci, (label, x_center) in enumerate(zip(cell_labels, [0.20, 0.50, 0.80])):
        micro_ax = ax.inset_axes([x_center-0.12, 0.08, 0.24, 0.32])
        micro_ax.axis('off')
        micro_ax.set_facecolor('white')

        # Same topology, different edge weights
        np.random.seed(ci * 10 + 5)
        edge_weights = np.random.uniform(0.4, 2.0, size=len(G_macro.edges()))
        edge_weights = edge_weights / edge_weights.max() * (1.5 + ci * 0.6)

        e_colors = [C['red'] if i%2==0 else C['blue'] for i in range(len(G_macro.edges()))]
        nx.draw_networkx_nodes(G_macro, macro_pos, ax=micro_ax, node_size=200,
                               node_color='#F3E3C3', edgecolors='#2B2B2B', linewidths=1.0)
        nx.draw_networkx_labels(G_macro, macro_pos, ax=micro_ax, font_size=6, font_weight='bold')
        nx.draw_networkx_edges(G_macro, macro_pos, ax=micro_ax, edge_color=e_colors,
                               width=edge_weights, arrowstyle='-|>', arrowsize=8,
                               connectionstyle='arc3,rad=0.1')
        micro_ax.set_title(label, fontsize=7, fontweight='bold', color=C['dark'], pad=1)

    # --- Hill kinetics formula (right side) ---
    ax.text(0.96, 0.72, r'$\mathbf{\mathcal{J}_{j,k}^{(i)} =}$', fontsize=9.5,
            fontweight='bold', color=C['dark'], transform=ax.transAxes, ha='right')
    ax.text(0.96, 0.64, r'$\mathbf{W_{global}^{(j,k)} \cdot}$', fontsize=9.5,
            fontweight='bold', color=C['dark'], transform=ax.transAxes, ha='right')
    ax.text(0.96, 0.56,
            r'$\mathbf{\left[ \frac{(X_{i,k})^n}{K_k^n + (X_{i,k})^n} \right]}$',
            fontsize=10, fontweight='bold', color=C['purple'], transform=ax.transAxes, ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#EDE7F6', edgecolor=C['purple'], alpha=0.7, lw=1.2))

    ax.text(0.96, 0.46, r'$n=2$ (dimerization)', fontsize=7.5, color=C['sub'],
            transform=ax.transAxes, ha='right')
    ax.text(0.96, 0.40, r'$K_k = \mathrm{median}(X_k[X_k>0])$', fontsize=7.5, color=C['sub'],
            transform=ax.transAxes, ha='right')

    # Output
    ax.text(0.5, 0.025,
            r'Output: $\boldsymbol{\mathcal{J}} \in \mathbb{R}^{N \times M \times M}$  (cell-specific regulatory activity tensor)',
            fontsize=7.5, ha='center', color=C['sub'], transform=ax.transAxes, fontstyle='italic')


# ============================================================
# MAIN ASSEMBLY
# ============================================================
def main():
    fig_w = PANEL_W * 2 + 0.8   # ≈ 13.6"
    fig_h = PANEL_H * 2 + 0.7   # ≈ 8.6"
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')

    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28,
                          left=0.04, right=0.99, top=0.97, bottom=0.03)

    ax_a = fig.add_subplot(gs[0, 0]); panel_a(ax_a)
    ax_b = fig.add_subplot(gs[0, 1]); panel_b(ax_b)
    ax_c = fig.add_subplot(gs[1, 0]); panel_c(ax_c)
    ax_d = fig.add_subplot(gs[1, 1]); panel_d(ax_d)

    fig.suptitle('TEG-Net: Transcriptional-noise Empowered Causal Network Inference Framework',
                 fontsize=14, fontweight='bold', y=0.993)

    for fmt, dpi in [('pdf', 600), ('png', 300)]:
        path = os.path.join(OUT_DIR, f'Figure_Methodology_Flow.{fmt}')
        plt.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"  [{fmt.upper()}] {path}")
    plt.close()
    print(f"\n[DONE] Figure: {fig_w:.1f}\"x{fig_h:.1f}\" | Panels: {PANEL_W:.1f}x{PANEL_H:.2f} (1:{PANEL_W/PANEL_H:.3f})")


if __name__ == "__main__":
    main()
