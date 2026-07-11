"""Synthetic end-to-end smoke test for scripts.melanoma_workflow (no network).

Exercises the full P4 wiring on tiny in-memory / file fixtures, including the
integration the P2/P3 reviews flagged: that a Tirosh-shaped AnnData (object-dtype
``cell_type``, ``.X`` in log space) flows through select_markers/build_signature.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest
from scripts.melanoma_workflow import (
    ScalingMethod,
    build_tumour_signature,
    correlate,
    cytotoxicity_score,
    harmonise_bulk,
    harmonize_expression_space,
    lineage_fraction,
    main,
    solver_diagnostics,
)

from scdecon.signature import SignatureConfig

# Marker genes per synthetic cell type; NK markers are the cytotoxic genes so a
# bulk mixture's cytotoxicity tracks its NK content.
_TYPE_GENES = {
    "malignant": ["MLANA", "PMEL"],
    "T": ["CD3D", "CD3E"],
    "NK": ["GZMA", "GZMB", "PRF1", "NKG7", "GZMH"],
    "B": ["MS4A1", "CD79A"],
}
_ALL_GENES = [gene for genes in _TYPE_GENES.values() for gene in genes]


def _synthetic_reference(*, categorical: bool = False) -> anndata.AnnData:
    """Log-space reference AnnData mimicking the Tirosh loader's output."""
    cell_types: list[str] = []
    rows: list[np.ndarray] = []
    rng = np.random.default_rng(0)
    gene_index = {gene: i for i, gene in enumerate(_ALL_GENES)}
    for cell_type, markers in _TYPE_GENES.items():
        for _ in range(6):
            linear = rng.random(len(_ALL_GENES)) * 0.1  # low background
            for gene in markers:
                linear[gene_index[gene]] = 50.0 + rng.normal(0.0, 1.0)
            rows.append(linear)
            cell_types.append(cell_type)
    linear_matrix = np.clip(np.asarray(rows, dtype=np.float64), 0.0, None)
    x_log1p = np.log1p(linear_matrix)  # .X is natural log1p, like the loader
    index = [f"cell{i}" for i in range(len(cell_types))]
    if categorical:
        obs = pd.DataFrame({"cell_type": pd.Categorical(cell_types)}, index=index)
    else:
        obs = pd.DataFrame({"cell_type": cell_types}, index=index)  # object dtype
    var = pd.DataFrame(index=_ALL_GENES)
    adata = anndata.AnnData(X=x_log1p, obs=obs, var=var)
    adata.layers["counts"] = linear_matrix
    return adata


def _synthetic_bulk_symbols(
    signature: pd.DataFrame,
) -> tuple[pd.DataFrame, list[float]]:
    """Build symbol-indexed bulk samples with increasing NK content."""
    cell_types = list(signature.columns)
    nk_index = cell_types.index("NK")
    t_index = cell_types.index("T")
    samples: dict[str, np.ndarray] = {}
    nk_fractions: list[float] = []
    for i in range(12):
        proportions = np.full(len(cell_types), 0.1)
        nk = 0.05 + i * 0.05
        proportions[nk_index] = nk
        proportions[t_index] = 0.2
        proportions = proportions / proportions.sum()
        mixture = signature.to_numpy() @ proportions  # genes-vector (linear)
        samples[f"s{i}"] = mixture * 1000.0
        nk_fractions.append(float(proportions[nk_index]))
    bulk = pd.DataFrame(samples, index=signature.index)
    return bulk, nk_fractions


def test_harmonise_bulk_maps_to_symbols() -> None:
    bulk = pd.DataFrame(
        {"s1": [10, 20], "s2": [0, 5]},
        index=pd.Index(["ENSG001.3", "ENSG002.7"], name="gene_id"),
    )
    gene_map = {"ENSG001": "TP53", "ENSG002": "EGFR"}
    bulk_symbols, coverage = harmonise_bulk(bulk, gene_map)
    assert list(bulk_symbols.index) == ["TP53", "EGFR"]
    assert coverage.n_mapped == 2
    assert coverage.fraction_mapped == 1.0


def test_harmonize_expression_space_scales_and_restricts_to_shared() -> None:
    signature = pd.DataFrame(
        {"A": [10.0, 0.0, 4.0], "B": [0.0, 20.0, 4.0]},
        index=pd.Index(["g1", "g2", "g3"], name="gene"),
    )
    bulk = pd.DataFrame(
        {"s1": [100.0, 200.0, 50.0], "s2": [10.0, 20.0, 5.0]},
        index=pd.Index(["g1", "g2", "gX"], name="gene"),  # g3 missing, gX extra
    )
    sig_h, bulk_h = harmonize_expression_space(signature, bulk, library_normalize=False)
    # Only shared genes (signature order); gX dropped, g3 dropped.
    assert list(sig_h.index) == ["g1", "g2"]
    assert list(bulk_h.index) == ["g1", "g2"]
    # Per-gene scale = signature row mean: g1 -> 5, g2 -> 10.
    np.testing.assert_allclose(sig_h.loc["g1"].to_numpy(), [10.0 / 5.0, 0.0 / 5.0])
    np.testing.assert_allclose(bulk_h.loc["g1"].to_numpy(), [100.0 / 5.0, 10.0 / 5.0])
    np.testing.assert_allclose(bulk_h.loc["g2"].to_numpy(), [200.0 / 10.0, 20.0 / 10.0])


def test_harmonize_expression_space_scaling_methods() -> None:
    signature = pd.DataFrame(
        {"A": [3.0, 0.0], "B": [0.0, 4.0]},
        index=pd.Index(["g1", "g2"], name="gene"),
    )
    bulk = pd.DataFrame({"s1": [6.0, 8.0]}, index=pd.Index(["g1", "g2"], name="gene"))
    # g1 row=[3,0]: mean=1.5, max=3, L2=3. g2 row=[0,4]: mean=2, max=4, L2=4.
    sig_mean, _ = harmonize_expression_space(
        signature, bulk, method=ScalingMethod.MEAN, library_normalize=False
    )
    np.testing.assert_allclose(sig_mean.loc["g1"].to_numpy(), [3.0 / 1.5, 0.0])
    sig_max, _ = harmonize_expression_space(
        signature, bulk, method=ScalingMethod.MAX, library_normalize=False
    )
    np.testing.assert_allclose(sig_max.loc["g1"].to_numpy(), [3.0 / 3.0, 0.0])
    sig_l2, _ = harmonize_expression_space(
        signature, bulk, method=ScalingMethod.L2, library_normalize=False
    )
    np.testing.assert_allclose(sig_l2.loc["g2"].to_numpy(), [0.0, 4.0 / 4.0])


def test_build_signature_accepts_object_dtype_cell_type() -> None:
    # The Tirosh loader yields object-dtype cell_type; the workflow must cope.
    reference = _synthetic_reference(categorical=False)
    assert not isinstance(reference.obs["cell_type"].dtype, pd.CategoricalDtype)
    config = SignatureConfig(cell_type_key="cell_type", n_markers_per_type=2)
    signature, markers = build_tumour_signature(reference, config)
    assert set(signature.columns) == {"malignant", "T", "NK", "B"}
    assert (signature.to_numpy() >= 0).all()
    assert all(len(genes) >= 1 for genes in markers.per_type.values())


def test_in_memory_pipeline_runs_and_is_biologically_sensible() -> None:
    reference = _synthetic_reference()
    config = SignatureConfig(cell_type_key="cell_type", n_markers_per_type=2)
    signature, _ = build_tumour_signature(reference, config)
    bulk_symbols, _ = _synthetic_bulk_symbols(signature)

    from scdecon.deconvolution import deconvolve

    proportions = deconvolve(signature, bulk_symbols, min_overlap=0.0)
    # Every sample's proportions sum to 1.
    np.testing.assert_allclose(proportions.sum(axis=0).to_numpy(), 1.0, atol=1e-9)

    diagnostics = solver_diagnostics(
        signature, bulk_symbols, proportions, min_overlap=0.0
    )
    assert diagnostics["n_samples_scored"] == bulk_symbols.shape[1]
    assert diagnostics["relative_residual_median"] < 0.2  # good fit on clean data

    score = cytotoxicity_score(bulk_symbols, ["GZMA", "GZMB", "PRF1", "NKG7", "GZMH"])
    tnk = lineage_fraction(proportions, ["T", "NK"])
    stats = correlate(tnk, score)
    assert stats["n"] == bulk_symbols.shape[1]
    # More NK -> more cytotoxicity and more T+NK: a clear positive correlation.
    assert stats["spearman_r"] > 0.5


def _write_reference(path: Path) -> None:
    header = "Cell\t" + "\t".join(f"c{i}" for i in range(8))
    tumor = "tumor\t" + "\t".join("1" for _ in range(8))
    # types: c0,c1 malignant; c2,c3 T; c4,c5 NK; c6,c7 B
    malignant = '"malignant(1=no,2=yes,0=unresolved)"\t2\t2\t1\t1\t1\t1\t1\t1'
    celltype = (
        '"non-malignant cell type (1=T,2=B,3=Macro,4=Endo,5=CAF,6=NK)"'
        "\t0\t0\t1\t1\t6\t6\t2\t2"
    )
    lines = [header, tumor, malignant, celltype]
    rng = np.random.default_rng(1)
    type_of_cell = ["malignant", "malignant", "T", "T", "NK", "NK", "B", "B"]
    for gene in _ALL_GENES:
        values = []
        for cell_type in type_of_cell:
            high = gene in _TYPE_GENES[cell_type]
            base = 5.0 if high else rng.random() * 0.1
            values.append(f"{base:.3f}")
        lines.append(gene + "\t" + "\t".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gtf(path: Path) -> dict[str, str]:
    ensembl = {gene: f"ENSG{1000 + i}" for i, gene in enumerate(_ALL_GENES)}
    lines = ["##tiny test GTF"]
    for gene, ens in ensembl.items():
        lines.append(
            f'chr1\tTEST\tgene\t1\t2\t.\t+\t.\tgene_id "{ens}.1"; gene_name "{gene}";'
        )
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return ensembl


def _write_gene_sums(path: Path, ensembl: dict[str, str]) -> None:
    header = "gene_id\t" + "\t".join(f"S{i}" for i in range(6))
    lines = ["##annotation=TEST", header]
    rng = np.random.default_rng(2)
    for gene in _ALL_GENES:
        counts = rng.integers(1, 100, size=6)
        # Make NK/cytotoxic genes vary across samples to drive a correlation.
        lines.append(f"{ensembl[gene]}.1\t" + "\t".join(str(int(c)) for c in counts))
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def test_main_writes_artefacts(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    reference_path = raw / "reference.txt"
    gtf_path = raw / "genes.gtf.gz"
    bulk_path = raw / "bulk.tsv.gz"
    _write_reference(reference_path)
    ensembl = _write_gtf(gtf_path)
    _write_gene_sums(bulk_path, ensembl)
    out_dir = tmp_path / "processed"

    rc = main(
        [
            "--bulk",
            str(bulk_path),
            "--gtf",
            str(gtf_path),
            "--reference",
            str(reference_path),
            "--out-dir",
            str(out_dir),
            "--n-markers",
            "2",
            "--min-overlap",
            "0.0",
        ]
    )
    assert rc == 0
    for name in (
        "proportions.tsv",
        "signature.tsv",
        "qc_report.json",
        "signature_heatmap.png",
        "tnk_vs_cytotoxicity.png",
    ):
        assert (out_dir / name).exists(), name

    report = json.loads((out_dir / "qc_report.json").read_text(encoding="utf-8"))
    assert report["mapping_coverage"]["n_mapped"] >= 1
    assert set(report["signature"]["cell_types"]) == {"malignant", "T", "NK", "B"}
    assert "relative_residual_median" in report["solver_diagnostics"]
    assert "primary_T_plus_NK_vs_cytotoxicity" in report["sanity_check"]
    # P4.1 expression-space diagnostics + before/after comparison.
    assert "normalization_applied" in report["expression_space_diagnostics"]
    before_after = report["before_after"]
    assert "before_raw_counts" in before_after
    assert "after_expression_space_harmonised" in before_after
    assert "T_plus_NK" in before_after["after_expression_space_harmonised"]


def test_cytotoxicity_score_requires_present_genes() -> None:
    bulk = pd.DataFrame({"s1": [1.0]}, index=pd.Index(["FOO"], name="gene"))
    with pytest.raises(ValueError, match="cytotoxicity genes"):
        cytotoxicity_score(bulk, ["GZMA", "GZMB"])
