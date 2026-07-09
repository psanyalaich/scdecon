"""Unit tests for scdecon.simulation.pseudobulk."""

from __future__ import annotations

import anndata
import numpy as np
import pandas as pd
import pytest

from scdecon.simulation import (
    BaseSimulator,
    CellSumSimulator,
    PseudobulkDataset,
    SimulationConfig,
    simulate_pseudobulk,
    split_reference,
)


def _profiles_adata() -> anndata.AnnData:
    """Three cell types; every cell of a type equals that type's profile."""
    profiles = {
        "A": [10.0, 0.0, 0.0, 1.0],
        "B": [0.0, 8.0, 0.0, 1.0],
        "C": [0.0, 0.0, 6.0, 1.0],
    }
    rows: list[list[float]] = []
    labels: list[str] = []
    for cell_type in ["A", "B", "C"]:
        for _ in range(5):
            rows.append(profiles[cell_type])
            labels.append(cell_type)
    counts = np.array(rows, dtype=np.float32)
    obs = pd.DataFrame(
        {"cell_type": pd.Categorical(labels)},
        index=[f"c{i}" for i in range(len(labels))],
    )
    var = pd.DataFrame(index=["g0", "g1", "g2", "g3"])
    return anndata.AnnData(X=counts, obs=obs, var=var)


def _profiles_matrix() -> np.ndarray:
    # genes x cell types, columns ordered by sorted cell types (A, B, C)
    return np.array(
        [[10.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 6.0], [1.0, 1.0, 1.0]]
    )


def _identical_adata(vector: list[float]) -> anndata.AnnData:
    counts = np.tile(np.array(vector, dtype=np.float32), (10, 1))
    labels = ["A"] * 5 + ["B"] * 5
    obs = pd.DataFrame(
        {"cell_type": pd.Categorical(labels)}, index=[f"c{i}" for i in range(10)]
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(len(vector))])
    return anndata.AnnData(X=counts, obs=obs, var=var)


def _config(**overrides: object) -> SimulationConfig:
    params: dict[str, object] = {
        "n_samples": 4,
        "n_cells_per_sample": 10,
        "cell_type_key": "cell_type",
        "random_state": 0,
    }
    params.update(overrides)
    return SimulationConfig(**params)  # type: ignore[arg-type]


# --- simulate_pseudobulk: shapes & invariants ------------------------------


def test_shapes_and_labels() -> None:
    dataset = simulate_pseudobulk(_profiles_adata(), _config())
    assert isinstance(dataset, PseudobulkDataset)
    assert dataset.bulk.shape == (4, 4)  # 4 genes x 4 samples
    assert dataset.proportions.shape == (3, 4)  # 3 cell types x 4 samples
    assert dataset.proportions.index.tolist() == ["A", "B", "C"]
    assert dataset.bulk.columns.tolist() == dataset.proportions.columns.tolist()


def test_realised_proportions_non_negative_and_sum_to_one() -> None:
    dataset = simulate_pseudobulk(_profiles_adata(), _config())
    values = dataset.proportions.to_numpy()
    assert (values >= 0).all()
    np.testing.assert_allclose(values.sum(axis=0), 1.0, rtol=1e-12)


def test_total_sampled_cells_equals_config() -> None:
    config = _config(n_cells_per_sample=10)
    dataset = simulate_pseudobulk(_profiles_adata(), config)
    # realised counts = proportions * N must be integers summing to N.
    counts = dataset.proportions.to_numpy() * config.n_cells_per_sample
    np.testing.assert_allclose(counts, np.round(counts), atol=1e-9)
    np.testing.assert_allclose(counts.sum(axis=0), config.n_cells_per_sample)


def test_pseudobulk_equals_sum_of_sampled_cells() -> None:
    """With identical cells, every sample must equal N * the cell vector."""
    vector = [2.0, 3.0, 5.0, 7.0]
    config = _config(n_cells_per_sample=10)
    dataset = simulate_pseudobulk(_identical_adata(vector), config)
    expected_column = config.n_cells_per_sample * np.array(vector)
    for sample in dataset.bulk.columns:
        np.testing.assert_allclose(dataset.bulk[sample].to_numpy(), expected_column)


def test_pseudobulk_is_proportion_weighted_profile_sum() -> None:
    """b_j = N * (profiles @ p_true_j): ties summation to the ground truth."""
    config = _config(n_cells_per_sample=10)
    dataset = simulate_pseudobulk(_profiles_adata(), config)
    expected = config.n_cells_per_sample * (
        _profiles_matrix() @ dataset.proportions.to_numpy()
    )
    np.testing.assert_allclose(dataset.bulk.to_numpy(), expected, rtol=1e-6)


def test_determinism() -> None:
    adata = _profiles_adata()
    first = simulate_pseudobulk(adata, _config())
    second = simulate_pseudobulk(adata, _config())
    pd.testing.assert_frame_equal(first.bulk, second.bulk)
    pd.testing.assert_frame_equal(first.proportions, second.proportions)


def test_counts_layer_is_used() -> None:
    adata = _identical_adata([1.0, 1.0, 1.0])
    adata.layers["counts"] = np.tile(
        np.array([2.0, 4.0, 6.0], dtype=np.float32), (10, 1)
    )
    from_layer = simulate_pseudobulk(adata, _config(counts_layer="counts"))
    from_x = simulate_pseudobulk(adata, _config(counts_layer=None))
    # layer has different values than X, so the sums differ.
    assert not np.allclose(from_layer.bulk.to_numpy(), from_x.bulk.to_numpy())
    np.testing.assert_allclose(
        from_layer.bulk["sample_0"].to_numpy(), 10 * np.array([2.0, 4.0, 6.0])
    )


# --- errors ----------------------------------------------------------------


def test_missing_cell_type_key_raises() -> None:
    with pytest.raises(ValueError, match="cell_type_key"):
        simulate_pseudobulk(_profiles_adata(), _config(cell_type_key="missing"))


def test_missing_counts_layer_raises() -> None:
    with pytest.raises(ValueError, match="counts_layer"):
        simulate_pseudobulk(_profiles_adata(), _config(counts_layer="absent"))


def test_simulator_can_be_used_directly() -> None:
    dataset = CellSumSimulator().simulate(_profiles_adata(), _config())
    assert isinstance(dataset, PseudobulkDataset)
    assert issubclass(CellSumSimulator, BaseSimulator)


# --- split_reference -------------------------------------------------------


def test_split_is_disjoint_and_covers_all() -> None:
    adata = _profiles_adata()
    signature, heldout = split_reference(adata)
    sig_names = set(signature.obs_names)
    held_names = set(heldout.obs_names)
    assert sig_names.isdisjoint(held_names)
    assert sig_names | held_names == set(adata.obs_names)


def test_split_is_stratified() -> None:
    adata = _profiles_adata()
    signature, heldout = split_reference(adata, signature_fraction=0.5)
    # every cell type appears in both partitions (5 cells each, 0.5 split)
    assert set(signature.obs["cell_type"].astype(str)) == {"A", "B", "C"}
    assert set(heldout.obs["cell_type"].astype(str)) == {"A", "B", "C"}


def test_split_is_deterministic() -> None:
    adata = _profiles_adata()
    first_sig, first_held = split_reference(adata, random_state=1)
    second_sig, second_held = split_reference(adata, random_state=1)
    assert first_sig.obs_names.tolist() == second_sig.obs_names.tolist()
    assert first_held.obs_names.tolist() == second_held.obs_names.tolist()


def test_split_fraction_controls_sizes() -> None:
    adata = _profiles_adata()  # 15 cells
    signature, heldout = split_reference(adata, signature_fraction=0.8)
    assert signature.n_obs == 12  # round(0.8 * 5) = 4 per type * 3 types
    assert heldout.n_obs == 3


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_split_invalid_fraction_raises(bad: float) -> None:
    with pytest.raises(ValueError, match="signature_fraction"):
        split_reference(_profiles_adata(), signature_fraction=bad)


def test_split_missing_key_raises() -> None:
    with pytest.raises(ValueError, match="cell_type_key"):
        split_reference(_profiles_adata(), cell_type_key="missing")
