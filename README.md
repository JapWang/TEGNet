# TEG-Net: Transcriptional-noise Empowered Causal Network

**TEG-Net** infers directed, signed gene regulatory networks (GRNs) from static single-cell RNA-seq data by exploiting **transcriptional heteroskedasticity** — the phenomenon that causal source genes exhibit lower residual variance than their targets.

Unlike pseudo-time or velocity-based methods, TEG-Net works on a single static expression cross-section, using asymmetry in HSIC (Hilbert-Schmidt Independence Criterion) scores to determine causal polarity without any time-series assumptions.

## Architecture

```
Expression Matrix (N cells × M genes)
    │
    ├─ [Module 1] DataPreprocessor: KNN imputation + MinMax scaling
    │
    ├─ [Module 2] HeteroskedasticCausalInferer: EBIC Graphical Lasso → HSIC asymmetry → Polarity matrix Π
    │
    ├─ [Module 3] STRREngine: Soft-mask Sequential Threshold Ridge Regression → W_global
    │
    ├─ [Module 4] OccupancyGating: Hill-function single-cell projection → 3D micro-tensor J
    │
    └─ [Module 5] NetworkVisualizer: Publication-quality figures
```

## Requirements

- Python ≥ 3.10

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run the Pipeline

```bash
python main.py
```

Edit the `__main__` block in `main.py` to point to your expression data:

```python
INPUT_FILE = r"path/to/your/expression.csv"
OUTPUT_DIR = r"path/to/output"
```

**Input format:** CSV with cells as rows and genes as columns. Values should be non-negative expression counts.

**Output files (in `OUTPUT_DIR`):**

| File | Description |
|------|-------------|
| `TEGNet_Polarity_Matrix_Pi.csv` | Causal polarity confidence matrix (M×M, row=target, col=source) |
| `TEGNet_Global_Macro_Network_W.csv` | Global signed regulatory network (M×M) |
| `TEGNet_Micro_Tensor_J.npy` | Cell-specific 3D regulatory tensor (N×M×M) |
| `Visualizations/` | PNG figures: polarity heatmap, HSIC evidence, network topology, dose-response curves |

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ebic_gamma` | 0.5 | EBIC sparsity penalty — higher = sparser network (0.5–2.0) |
| `knn_k` | 15 | KNN receptive field for local variance estimation (10–30) |
| `n_hill` | 2.0 | Hill cooperativity coefficient (2.0 = dimerization) |
| `poly_degree` | 1 | Polynomial degree for partial residual extraction (1 = linear, recommended) |

## Evaluation

After running the pipeline, evaluate the inferred network against a known ground truth:

```bash
python Evaluator.py
```

Edit `Evaluator.py`'s `__main__` block to set your ground truth and TEG-Net output paths.

## Visualization

Generate a methodology workflow figure:

```bash
python figure_methodology_flow.py
```

## License

This project is licensed under the MIT License.
