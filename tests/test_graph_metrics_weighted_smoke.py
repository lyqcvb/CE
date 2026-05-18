from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from step04_graph_metrics import (
    clustering_coef_wu,
    global_efficiency_weighted,
    graph_qc_metrics,
)
from step06_rich_club import compute_weighted_normalized_rich_club, weighted_rich_club


def test_weighted_graph_metrics_return_finite_values() -> None:
    weight = np.array(
        [
            [0.0, 0.9, 0.5, 0.0],
            [0.9, 0.0, 0.7, 0.3],
            [0.5, 0.7, 0.0, 0.2],
            [0.0, 0.3, 0.2, 0.0],
        ],
        dtype=np.float64,
    )

    eglob = global_efficiency_weighted(weight)
    clustering = clustering_coef_wu(weight)
    richclub = weighted_rich_club(weight)

    assert np.isfinite(eglob)
    assert eglob > 0.0
    assert clustering.shape == (4,)
    assert np.isfinite(clustering).all()
    assert richclub.ndim == 1


def test_graph_qc_metrics_reports_density_and_components() -> None:
    graph = np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=np.float64,
    )

    qc = graph_qc_metrics(graph)

    assert qc["EdgeCount"] == 2.0
    assert qc["ComponentCount"] == 2.0
    assert qc["LargestComponentSize"] == 2.0
    np.testing.assert_allclose(qc["Density"], 2.0 / 6.0)


def test_weighted_rich_club_random_normalization_has_expected_fields() -> None:
    weight = np.array(
        [
            [0.0, 0.9, 0.5, 0.0],
            [0.9, 0.0, 0.7, 0.3],
            [0.5, 0.7, 0.0, 0.2],
            [0.0, 0.3, 0.2, 0.0],
        ],
        dtype=np.float64,
    )

    stats = compute_weighted_normalized_rich_club(weight, n_rand=5, seed=1)

    expected = {
        "Phi_w_observed",
        "Phi_w_ranknorm",
        "Phi_w_random_mean",
        "Phi_w_random_std",
        "Phi_w_norm",
        "Phi_w_empirical_p",
        "Phi_w_random_valid_n",
    }
    assert set(stats) == expected
    lengths = {len(values) for values in stats.values()}
    assert len(lengths) == 1
