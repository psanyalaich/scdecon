# API reference

scdecon is a layered package with a strict, test-enforced dependency direction.
The pages below document each public module; the computational core
(`io`, `preprocessing`, `signature`, `deconvolution`) never imports the plotting
stack, and the numerical solvers operate on plain NumPy arrays.

| Module | Responsibility |
|--------|----------------|
| [`scdecon.io`](io.md) | Read/write `.h5ad`, bulk, and metadata tables (faithful, fixed orientation). |
| [`scdecon.preprocessing`](preprocessing.md) | QC filtering and normalisation, config-driven. |
| [`scdecon.signature`](signature.md) | Marker selection and signature-matrix construction. |
| [`scdecon.deconvolution`](deconvolution.md) | The `Solver` interface, NNLS/ν-SVR/robust solvers, alignment, `deconvolve`, and benchmarking. |
| [`scdecon.simulation`](simulation.md) | Pseudobulk simulation with known ground truth. |
| [`scdecon.validation`](validation.md) | Accuracy metrics (RMSE / Pearson / Spearman). |
| [`scdecon.genes`](genes.md) | Generic gene-identifier harmonisation. |
| [`scdecon.plotting`](plotting.md) | Figures (signature heatmap, truth-vs-prediction, benchmark). |

The command-line interface is documented separately in the
[CLI reference](../cli.md).
