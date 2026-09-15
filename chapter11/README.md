# Chapter 11 notebooks

Most of Chapter 11 runs in the book's main environment (`pyproject.toml` /
`ml_timeseries`):

- `Time Series Classification.ipynb`
- `Drift Detection.ipynb`
- `Causal Inference.ipynb`

Two notebooks need extra dependencies that are **not** in the main environment:

- `Graph WaveNet.ipynb` — a from-scratch Graph WaveNet for spatio-temporal
  forecasting. Needs `torch`.
- `Neural ODEs.ipynb` — a Neural ODE fitted to irregularly sampled data. Needs
  `torch` and `torchdiffeq`.

`torch` and `torchdiffeq` are deliberately kept out of `pyproject.toml`: they
pull in a large dependency tree, and these two notebooks are the only place in
the book that uses them. Run them in a dedicated environment instead.

## Dedicated environment

```bash
conda create -n ts_advanced python=3.11 -y
conda activate ts_advanced
pip install "numpy<2" torch torchdiffeq matplotlib jupyterlab
```

The `numpy<2` pin is required: the `torch` 2.2.x wheels are built against
numpy 1.x, and importing `torch` with numpy 2.x fails with
`Failed to initialize NumPy`. This is the same constraint documented in the
repository's top-level `README.md`.

Launch the two notebooks from this environment:

```bash
conda activate ts_advanced
jupyter lab "Graph WaveNet.ipynb" "Neural ODEs.ipynb"
```

Both run on CPU in a couple of minutes each; no GPU is needed.
