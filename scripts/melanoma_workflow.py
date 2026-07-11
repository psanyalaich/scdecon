"""End-to-end melanoma TME deconvolution workflow (M7 P4).

Dataset-specific **orchestration** (lives in ``scripts/``, not the package). It
wires the already-built, generic components into the real biological workflow:

    load recount3 bulk + GENCODE GTF + Tirosh scRNA reference
      -> harmonise gene IDs (scdecon.genes)              [GeneMappingCoverage QC]
      -> build tumour signature (scdecon.signature)
      -> deconvolve TCGA-SKCM (scdecon.deconvolution)
      -> cytotoxicity sanity check (T+NK fraction vs cytotoxicity signature)
      -> QC report + figures + proportions

Nothing here modifies the package; every numerical/biological primitive comes
from ``scdecon`` or the ``scripts.datasets`` loaders. Cross-platform caveats
(recount3 coverage counts vs Smart-seq2 TPM) mean results are **relative**
composition estimates, not absolute calibrated fractions (ADR-0010, D3/D6).

Run as a module from the repo root (so the ``scripts`` package resolves)::

    python -m scripts.melanoma_workflow          # uses data/raw + writes data/processed
    python -m scripts.melanoma_workflow --n-markers 50 --min-overlap 0.3
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import stats

from scdecon.deconvolution import align_signature_and_bulk, deconvolve
from scdecon.genes import (
    GeneAggregation,
    GeneMappingCoverage,
    compute_mapping_coverage,
    relabel_gene_index,
    strip_ensembl_version,
)
from scdecon.logging_utils import configure_logging, get_logger
from scdecon.plotting import plot_signature_heatmap
from scdecon.signature import (
    MarkerSet,
    SignatureConfig,
    build_signature,
    select_markers,
)
from scripts.datasets.annotations import parse_gtf_gene_map
from scripts.datasets.recount3 import load_recount3_bulk
from scripts.datasets.tirosh import load_tirosh_reference
from scripts.download_data import SOURCES

logger = get_logger("scripts.melanoma_workflow")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_DIR = _REPO_ROOT / "data"
_FILENAMES = {source.name: source.filename for source in SOURCES}

#: Primary cytotoxicity signature (D7). KLRD1 is a sensitivity-only add-on.
PRIMARY_CYTOTOXICITY_GENES: tuple[str, ...] = ("GZMA", "GZMB", "PRF1", "NKG7", "GZMH")
SENSITIVITY_GENES: tuple[str, ...] = ("KLRD1",)
#: Cytotoxic lineage for the primary sanity check (D8): T + NK combined.
CYTOTOXIC_LINEAGE: tuple[str, ...] = ("T", "NK")


class ScalingMethod(StrEnum):
    """Per-gene scaling factor used by :func:`harmonize_expression_space`.

    ``MEAN`` performed best experimentally on TCGA-SKCM and is the default;
    ``MAX`` and ``L2`` were part of the investigation and are retained for future
    datasets (all three restored the immune signal; see the tutorial).
    """

    MEAN = "mean"
    MAX = "max"
    L2 = "l2"


def harmonise_bulk(
    bulk: pd.DataFrame,
    gene_map: Mapping[str, str],
    *,
    min_coverage: float = 0.5,
) -> tuple[pd.DataFrame, GeneMappingCoverage]:
    """Map an Ensembl-indexed bulk matrix onto gene symbols.

    Ensembl version suffixes are stripped, coverage of the mapping is measured
    (:class:`~scdecon.genes.GeneMappingCoverage`), and rows are relabelled to
    symbols, summing any many-to-one collisions (correct for raw counts).

    Returns
    -------
    tuple[pandas.DataFrame, GeneMappingCoverage]
        The symbol-indexed bulk matrix and the mapping-coverage QC metric.
    """
    stripped = bulk.copy()
    stripped.index = pd.Index(
        [strip_ensembl_version(str(gene)) for gene in bulk.index],
        name=bulk.index.name,
    )
    coverage = compute_mapping_coverage(stripped.index, gene_map)
    bulk_symbols = relabel_gene_index(
        stripped,
        gene_map,
        aggregate=GeneAggregation.SUM,
        min_coverage=min_coverage,
    )
    return bulk_symbols, coverage


def build_tumour_signature(
    reference: anndata.AnnData, config: SignatureConfig
) -> tuple[pd.DataFrame, MarkerSet]:
    """Select markers and build the signature from a reference AnnData.

    Scanpy's marker ranking expects a categorical grouping column; the Tirosh
    loader yields an object-dtype ``cell_type``, so it is coerced to categorical
    here (a scripts-level adaptation; the package is unchanged).
    """
    cell_type_key = config.cell_type_key
    if not isinstance(reference.obs[cell_type_key].dtype, pd.CategoricalDtype):
        reference.obs[cell_type_key] = reference.obs[cell_type_key].astype("category")
    markers = select_markers(reference, config)
    signature = build_signature(reference, markers, config)
    return signature, markers


def solver_diagnostics(
    signature: pd.DataFrame,
    bulk_symbols: pd.DataFrame,
    proportions: pd.DataFrame,
    *,
    min_overlap: float = 0.0,
) -> dict[str, float | int]:
    """Per-sample fit diagnostics for a deconvolution result.

    For each sample, the optimal non-negative scale ``a`` minimising
    ``||b - a * (S @ p)||`` is found and the **relative residual**
    ``||b - a*(S @ p)|| / ||b||`` recorded. This is scale-free (the solver
    renormalises ``p``), so it is comparable across samples and flags samples the
    signature explains poorly.
    """
    aligned = align_signature_and_bulk(signature, bulk_symbols, min_overlap=min_overlap)
    s_matrix = aligned.signature
    b_matrix = aligned.bulk
    proportions_matrix = proportions.loc[
        aligned.cell_types, aligned.sample_names
    ].to_numpy(dtype=np.float64)
    predicted = s_matrix @ proportions_matrix

    relative_residuals: list[float] = []
    for column in range(b_matrix.shape[1]):
        observed = b_matrix[:, column]
        modelled = predicted[:, column]
        observed_norm = float(np.linalg.norm(observed))
        if observed_norm == 0.0:
            continue
        denominator = float(modelled @ modelled)
        scale = float(observed @ modelled) / denominator if denominator > 0.0 else 0.0
        residual = float(np.linalg.norm(observed - scale * modelled))
        relative_residuals.append(residual / observed_norm)

    residuals = np.asarray(relative_residuals, dtype=np.float64)
    return {
        "n_overlap_genes": len(aligned.genes),
        "overlap_fraction": len(aligned.genes) / signature.shape[0],
        "n_samples_scored": int(residuals.size),
        "relative_residual_median": (
            float(np.median(residuals)) if residuals.size else float("nan")
        ),
        "relative_residual_mean": (
            float(residuals.mean()) if residuals.size else float("nan")
        ),
        "relative_residual_p95": (
            float(np.percentile(residuals, 95)) if residuals.size else float("nan")
        ),
        "relative_residual_max": (
            float(residuals.max()) if residuals.size else float("nan")
        ),
    }


def cytotoxicity_score(bulk_symbols: pd.DataFrame, genes: Sequence[str]) -> pd.Series:
    """Per-sample cytotoxicity score = mean log-CPM over the given gene set.

    The bulk is CPM-normalised (per-sample library size) so scores are comparable
    across patients regardless of sequencing depth.

    Raises
    ------
    ValueError
        If none of ``genes`` are present in the bulk.
    """
    present = [gene for gene in genes if gene in bulk_symbols.index]
    if not present:
        raise ValueError(
            f"None of the cytotoxicity genes {list(genes)} are present in the bulk."
        )
    library = bulk_symbols.sum(axis=0)
    cpm = bulk_symbols.div(library, axis=1) * 1e6
    subset = cpm.loc[present]
    log_cpm = np.log1p(subset.to_numpy(dtype=np.float64))
    return pd.Series(log_cpm.mean(axis=0), index=subset.columns, name="cytotoxicity")


def lineage_fraction(proportions: pd.DataFrame, cell_types: Sequence[str]) -> pd.Series:
    """Per-sample summed fraction over the given cell types.

    Raises
    ------
    ValueError
        If none of ``cell_types`` are present in ``proportions``.
    """
    present = [cell_type for cell_type in cell_types if cell_type in proportions.index]
    if not present:
        raise ValueError(
            f"None of the cell types {list(cell_types)} are in the proportions."
        )
    fraction = proportions.loc[present].sum(axis=0)
    fraction.name = "+".join(present)
    return fraction


def correlate(x: pd.Series, y: pd.Series) -> dict[str, float | int]:
    """Spearman and Pearson correlation of two aligned per-sample series."""
    joined = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    n = int(len(joined))
    if n < 3:
        return {
            "n": n,
            "spearman_r": float("nan"),
            "spearman_p": float("nan"),
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
        }
    spearman = stats.spearmanr(joined["x"], joined["y"])
    pearson = stats.pearsonr(joined["x"], joined["y"])
    return {
        "n": n,
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
    }


def _per_gene_scale(signature: pd.DataFrame, method: ScalingMethod) -> pd.Series:
    """Per-gene scaling factor (one per gene) under the chosen method."""
    if method is ScalingMethod.MEAN:
        return signature.mean(axis=1)
    if method is ScalingMethod.MAX:
        return signature.max(axis=1)
    if method is ScalingMethod.L2:
        norms = np.linalg.norm(signature.to_numpy(dtype=np.float64), axis=1)
        return pd.Series(norms, index=signature.index)
    raise ValueError(f"Unknown scaling method: {method!r}")


def harmonize_expression_space(
    signature: pd.DataFrame,
    bulk: pd.DataFrame,
    *,
    method: ScalingMethod = ScalingMethod.MEAN,
    library_normalize: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Put the bulk and signature into a comparable per-gene expression space.

    Diagnosed in P4.1 on the real data: deconvolving raw recount3 counts against
    the TPM signature collapses the small immune populations because the NNLS
    least-squares fit is dominated by a handful of very high-magnitude genes.
    Gene-length (TPM) normalisation was tested and did **not** resolve this, so
    the evidence is most consistent with per-gene magnitude dominance rather than
    gene length. This transform removes that dominance:

    1. optionally library-normalise the bulk to CPM (per-sample; this is neutral
       for the scale-invariant NNLS solver but makes magnitudes interpretable);
    2. divide every gene row of **both** the signature and the bulk by a per-gene
       scale derived from the signature (``method``: mean/max/L2 across cell
       types).

    **Why this is model-preserving.** Let ``D`` be the diagonal matrix of per-gene
    scaling factors. The transform is a left multiplication by the invertible
    ``D**-1``: ``S' = D**-1 S`` and ``B' = D**-1 B``. The linear mixture model
    ``B ~= S p`` becomes ``D**-1 B ~= D**-1 S p`` -- i.e. ``B' ~= S' p`` with the
    **same** unknown proportions ``p``. So the biological model is unchanged; only
    the per-gene *weighting* of the least-squares objective changes, letting every
    marker gene contribute comparably instead of a few high-magnitude genes
    dominating.

    Parameters
    ----------
    signature:
        Signature matrix (genes x cell types).
    bulk:
        Bulk matrix (genes x samples), gene-indexed on the same identifiers.
    method:
        Per-gene scaling factor (:class:`ScalingMethod`); ``MEAN`` by default.
    library_normalize:
        If ``True`` (default), CPM-normalise the bulk per sample first.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        The harmonised ``(signature, bulk)``, restricted to their shared genes in
        signature row order.

    Raises
    ------
    ValueError
        If the signature and bulk share no genes.
    """
    bulk_genes = set(bulk.index)
    shared = [gene for gene in signature.index if gene in bulk_genes]
    if not shared:
        raise ValueError("Signature and bulk share no genes for harmonisation.")

    signature_shared = signature.loc[shared]
    if library_normalize:
        bulk = bulk.div(bulk.sum(axis=0), axis=1) * 1e6
    bulk_shared = bulk.loc[shared]

    scale = _per_gene_scale(signature_shared, method).replace(0.0, np.nan)
    signature_h = signature_shared.div(scale, axis=0).fillna(0.0)
    bulk_h = bulk_shared.div(scale, axis=0).fillna(0.0)
    return signature_h, bulk_h


def _distribution(series: pd.Series) -> dict[str, float]:
    """Min/median/mean/max summary of a per-sample series."""
    return {
        "min": float(series.min()),
        "median": float(series.median()),
        "mean": float(series.mean()),
        "max": float(series.max()),
    }


def summarise_run(
    signature: pd.DataFrame,
    bulk: pd.DataFrame,
    proportions: pd.DataFrame,
    score: pd.Series,
) -> dict[str, object]:
    """Collect the P4.1 before/after comparison metrics for one deconvolution.

    ``signature``/``bulk`` are whatever were passed to the solver (raw or
    harmonised), so the residual is measured in that run's own fitting space and
    is only loosely comparable across spaces — the decisive metrics are the immune
    fractions' dynamic range and their correlation with the cytotoxicity score.
    """
    diagnostics = solver_diagnostics(signature, bulk, proportions, min_overlap=0.0)
    tnk = lineage_fraction(proportions, CYTOTOXIC_LINEAGE)
    t_only = lineage_fraction(proportions, ("T",))
    nk_only = lineage_fraction(proportions, ("NK",))
    return {
        "median_relative_residual": diagnostics["relative_residual_median"],
        "mean_fraction": {
            str(cell_type): float(value)
            for cell_type, value in proportions.mean(axis=1).items()
        },
        "T_plus_NK": _distribution(tnk),
        "T_only": _distribution(t_only),
        "NK_only": _distribution(nk_only),
        "spearman_T_plus_NK": correlate(tnk, score)["spearman_r"],
        "pearson_T_plus_NK": correlate(tnk, score)["pearson_r"],
        "spearman_T_only": correlate(t_only, score)["spearman_r"],
        "spearman_NK_only": correlate(nk_only, score)["spearman_r"],
    }


def plot_sanity_scatter(
    fraction: pd.Series,
    score: pd.Series,
    path: str | Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    annotation: str | None = None,
) -> Path:
    """Scatter of predicted fraction vs a bulk signature score (headless-safe)."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    joined = pd.concat([fraction.rename("f"), score.rename("s")], axis=1).dropna()
    figure = Figure(figsize=(6.0, 5.0))
    axes = figure.subplots()
    axes.scatter(joined["f"], joined["s"], s=14, alpha=0.6)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    if annotation is not None:
        axes.annotate(
            annotation,
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            va="top",
            fontsize=9,
        )
    figure.tight_layout()
    figure.savefig(resolved, dpi=150)
    logger.info("Wrote %s", resolved)
    return resolved


def _install_warning_capture() -> list[str]:
    """Capture WARNING+ records from the scdecon logger for the QC report."""
    captured: list[str] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(f"{record.name}: {record.getMessage()}")

    handler = _ListHandler(level=logging.WARNING)
    logging.getLogger("scdecon").addHandler(handler)
    return captured


def main(argv: list[str] | None = None) -> int:
    """Run the full workflow and write proportions, a QC report, and figures."""
    configure_logging()
    parser = argparse.ArgumentParser(description="Melanoma TME deconvolution workflow.")
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--bulk", type=Path, default=None)
    parser.add_argument("--gtf", type=Path, default=None)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--n-markers", type=int, default=25)
    parser.add_argument("--min-overlap", type=float, default=0.5)
    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir
    raw = data_dir / "raw"
    bulk_path: Path = args.bulk or raw / _FILENAMES["tcga_gene_sums"]
    gtf_path: Path = args.gtf or raw / _FILENAMES["gencode_gtf"]
    reference_path: Path = args.reference or raw / _FILENAMES["tirosh_scrna"]
    out_dir: Path = args.out_dir or data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_markers: int = args.n_markers
    min_overlap: float = args.min_overlap

    captured_warnings = _install_warning_capture()

    logger.info("Loading inputs ...")
    bulk = load_recount3_bulk(bulk_path)
    gene_map = parse_gtf_gene_map(gtf_path)
    reference = load_tirosh_reference(reference_path)

    logger.info("Harmonising bulk gene identifiers ...")
    bulk_symbols, coverage = harmonise_bulk(bulk, gene_map)
    logger.info("Gene-ID mapping coverage: %s", coverage.render())

    logger.info("Building tumour signature ...")
    config = SignatureConfig(cell_type_key="cell_type", n_markers_per_type=n_markers)
    signature, markers = build_tumour_signature(reference, config)
    plot_signature_heatmap(
        signature, out_dir / "signature_heatmap.png", title="Melanoma TME signature"
    )

    primary_score = cytotoxicity_score(bulk_symbols, PRIMARY_CYTOTOXICITY_GENES)

    # Before: deconvolve the raw harmonised-ID counts (the P4 baseline).
    logger.info(
        "Deconvolving (baseline: raw counts) %d samples ...", bulk_symbols.shape[1]
    )
    proportions_before = deconvolve(signature, bulk_symbols, min_overlap=min_overlap)
    before = summarise_run(signature, bulk_symbols, proportions_before, primary_score)

    # After: harmonise the expression space, then deconvolve (the primary result).
    logger.info("Harmonising expression space and re-deconvolving ...")
    scaling_method = ScalingMethod.MEAN
    signature_h, bulk_h = harmonize_expression_space(
        signature, bulk_symbols, method=scaling_method
    )
    proportions = deconvolve(signature_h, bulk_h, min_overlap=min_overlap)
    after = summarise_run(signature_h, bulk_h, proportions, primary_score)

    diagnostics = solver_diagnostics(signature_h, bulk_h, proportions)
    tnk_fraction = lineage_fraction(proportions, CYTOTOXIC_LINEAGE)
    primary_stats = correlate(tnk_fraction, primary_score)
    sanity: dict[str, object] = {
        "primary_T_plus_NK_vs_cytotoxicity": primary_stats,
        "secondary_T_only": correlate(
            lineage_fraction(proportions, ("T",)), primary_score
        ),
        "secondary_NK_only": correlate(
            lineage_fraction(proportions, ("NK",)), primary_score
        ),
        "sensitivity_with_KLRD1": correlate(
            tnk_fraction,
            cytotoxicity_score(
                bulk_symbols, (*PRIMARY_CYTOTOXICITY_GENES, *SENSITIVITY_GENES)
            ),
        ),
        "cytotoxicity_genes": list(PRIMARY_CYTOTOXICITY_GENES),
        "lineage": list(CYTOTOXIC_LINEAGE),
    }

    plot_sanity_scatter(
        tnk_fraction,
        primary_score,
        out_dir / "tnk_vs_cytotoxicity.png",
        title="T+NK fraction vs cytotoxicity signature (TCGA-SKCM)",
        xlabel="Predicted T+NK fraction",
        ylabel="Cytotoxicity score (mean log-CPM)",
        annotation=(
            f"Spearman r={primary_stats['spearman_r']:.2f} (n={primary_stats['n']})"
        ),
    )

    qc_report: dict[str, object] = {
        "n_samples": int(proportions.shape[1]),
        "mapping_coverage": coverage.to_dict(),
        "signature": {
            "n_marker_genes": int(signature.shape[0]),
            "cell_types": [str(column) for column in signature.columns],
            "markers_per_type": {
                cell_type: len(genes) for cell_type, genes in markers.per_type.items()
            },
        },
        "cohort_composition_mean": {
            str(cell_type): float(value)
            for cell_type, value in proportions.mean(axis=1).items()
        },
        "solver_diagnostics": diagnostics,
        "expression_space_diagnostics": {
            "bulk_scale": "recount3 exonic coverage counts",
            "reference_scale": (
                "Tirosh log2(TPM/10+1) -> linear TPM; signature = mean linear TPM"
            ),
            "normalization_applied": (
                "library (CPM) + per-gene relative scaling (left multiplication by "
                "an invertible diagonal D**-1 applied to both signature and bulk, "
                "which preserves the model B ~= S*p and only reweights the "
                "least-squares objective)"
            ),
            "scaling_method": scaling_method.value,
            "length_normalization_evaluated": (
                "gene-length (TPM) normalisation, as implemented here, did not "
                "resolve the collapse (residual ~0.70 -> ~0.62, T/NK still ~0); "
                "this does not exclude alternative formulations or other "
                "contributing factors"
            ),
            "expected_compatibility": (
                "moderate (relative estimates; cross-platform; NK vs CD8-T not "
                "separately identifiable, so the combined T+NK fraction is used)"
            ),
        },
        "before_after": {
            "before_raw_counts": before,
            "after_expression_space_harmonised": after,
        },
        "sanity_check": sanity,
        "captured_warnings": captured_warnings,
        "params": {
            "n_markers_per_type": n_markers,
            "min_overlap": min_overlap,
            "solver": "NNLSSolver",
        },
        "notes": [
            "Primary results use expression-space-harmonised bulk (see "
            "expression_space_diagnostics); the raw-count baseline is retained in "
            "before_after for comparison.",
            "Results are relative composition estimates, not absolute calibrated "
            "fractions (cross-platform; D3).",
            "The Tirosh reference does not distinguish CD8 from CD4 T cells and NK "
            "shares cytotoxic markers with CD8-T; the primary sanity check uses the "
            "combined T+NK fraction (D8).",
        ],
    }

    (out_dir / "proportions.tsv").write_text(
        proportions.to_csv(sep="\t"), encoding="utf-8"
    )
    (out_dir / "signature.tsv").write_text(signature.to_csv(sep="\t"), encoding="utf-8")
    (out_dir / "qc_report.json").write_text(
        json.dumps(qc_report, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Wrote artefacts to %s", out_dir)
    logger.info("Primary sanity check (T+NK vs cytotoxicity): %s", primary_stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
