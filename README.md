# How Accurately Does a Transformer Store Integer Intervals?

A mechanistic interpretability study investigating how GPT-2 small internally represents integer intervals, number comparisons, and modular arithmetic (alphabetic ring counting) in its residual stream.

## Key Findings

- **Comparison information is nearly perfectly encoded internally** (99.1% probe accuracy at layer 9) despite only 48.6% behavioral accuracy — a massive representation-behavior gap
- **Interval membership is harder than single comparison**: peaks at 83.8% probe accuracy (layer 8), ~16pp below comparison, consistent with needing two boundary checks
- **Ring counting is computed internally but never output**: 58% probe accuracy at layer 12 vs. 0% behavioral accuracy — the model computes modular arithmetic but cannot express it
- **Numerical values are perfectly encoded** from layer 1 onward (R² > 0.99), confirming prior work on number representation
- **Layers 8-9 are causally important** for both comparison and interval tasks (activation patching), matching known greater-than circuit location

## Project Structure

```
├── REPORT.md                  # Full research report with results
├── planning.md                # Research plan and methodology
├── literature_review.md       # Literature review
├── resources.md               # Resource catalog
├── src/
│   ├── run_experiments.py     # Main experiment runner
│   ├── visualize.py           # Visualization generation
│   ├── data_generation.py     # Dataset generation utilities
│   └── experiments.py         # Extended experiment code
├── results/
│   ├── experiment_results.json  # All numerical results
│   └── figures/                 # Generated plots
│       ├── behavioral_accuracy.png
│       ├── probing_by_layer.png
│       ├── regression_probing.png
│       ├── width_analysis.png
│       ├── activation_patching.png
│       └── combined_probing.png
├── papers/                    # Downloaded reference papers
├── datasets/                  # Sample datasets
└── code/                      # Cloned reference repositories
```

## Reproducing Results

```bash
# Setup
uv venv && source .venv/bin/activate
uv add torch numpy matplotlib scikit-learn tqdm transformer-lens
uv pip install 'transformers<5'  # compatibility fix

# Run experiments (~2 minutes on RTX A6000)
cd src && python run_experiments.py

# Generate visualizations
python visualize.py
```

**Requirements**: Python 3.10+, CUDA-capable GPU (tested on RTX A6000, 49GB), ~2GB disk for model weights.

## Method

1. **Behavioral evaluation**: Measure GPT-2's accuracy on comparison, interval membership, and ring counting tasks
2. **Linear probing**: Train linear classifiers on residual stream activations at each layer to decode task labels
3. **Regression probing**: Decode numerical values (X, offset N) from hidden states via Ridge regression
4. **Activation patching**: Swap residual stream between positive/negative examples to identify causally important layers

See [REPORT.md](REPORT.md) for full methodology, results, and analysis.
