from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from step02_connectivity import compute_hofc, compute_lofc, proportional_threshold
from step03_connection_strength import average_connection_strength


def test_average_connection_strength_matches_undirected_upper_triangle() -> None:
    matrix = np.array(
        [
            [1.0, 0.2, 0.4],
            [0.2, 1.0, 0.6],
            [0.4, 0.6, 1.0],
        ],
        dtype=np.float64,
    )

    mean_conn = average_connection_strength(matrix, directed=False)

    np.testing.assert_allclose(mean_conn, 0.4)


def test_average_connection_strength_matches_directed_off_diagonal() -> None:
    matrix = np.array(
        [
            [0.0, 0.1, 0.2],
            [0.3, 0.0, 0.4],
            [0.5, 0.6, 0.0],
        ],
        dtype=np.float64,
    )

    mean_conn = average_connection_strength(matrix, directed=True)

    np.testing.assert_allclose(mean_conn, 0.35)


def test_lofc_hofc_smoke_shapes_and_finite_values() -> None:
    t = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    eeg = np.vstack(
        [
            np.sin(t),
            np.sin(t + 0.3),
            np.cos(2.0 * t),
            np.sin(3.0 * t + 0.7),
        ]
    )

    lofc = compute_lofc(eeg)
    hofc = compute_hofc(lofc)

    assert lofc.shape == (4, 4)
    assert hofc.shape == (4, 4)
    assert np.isfinite(lofc).all()
    assert np.isfinite(hofc).all()
    np.testing.assert_allclose(np.diag(lofc), 1.0)
    np.testing.assert_allclose(hofc, hofc.T)


def test_proportional_threshold_keeps_strongest_undirected_edges() -> None:
    matrix = np.array(
        [
            [0.0, 0.9, 0.1, 0.4],
            [0.9, 0.0, 0.8, 0.2],
            [0.1, 0.8, 0.0, 0.7],
            [0.4, 0.2, 0.7, 0.0],
        ],
        dtype=np.float64,
    )

    graph = proportional_threshold(matrix, density=0.5, directed=False)

    assert graph.sum() == 6.0
    assert graph[0, 1] == graph[1, 0] == 1.0
    assert graph[1, 2] == graph[2, 1] == 1.0
    assert graph[2, 3] == graph[3, 2] == 1.0
