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

    assert np.isfinite(eglob)
    assert eglob > 0.0
    assert clustering.shape == (4,)
    assert np.isfinite(clustering).all()


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
