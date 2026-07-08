# scdecon

*Single-cell-reference deconvolution of bulk tumour transcriptomes.*

Bulk RNA-seq measures the **average** expression of a whole tissue — a tumour
biopsy is a blend of cancer cells, T cells, B cells, macrophages, fibroblasts,
and more. `scdecon` estimates **what fraction of a bulk sample comes from each
cell type**, using an annotated single-cell RNA-seq atlas as the reference.

It builds a cell-type signature matrix from single-cell data, then solves a
constrained regression to infer cell-type proportions in bulk samples —
`Bulk ≈ Signature × Proportions`, subject to non-negativity and sum-to-one.

## Installation

Requires Python 3.11+.

```bash
git clone https://github.com/psanyalaich/scdecon
cd scdecon
pip install -e ".[dev]"
```

## Development

```bash
pytest          # run tests
ruff check .    # lint
ruff format .   # format
mypy            # type-check
```

## License

[MIT](LICENSE) © 2026 Prisha Sanyal-Aich
