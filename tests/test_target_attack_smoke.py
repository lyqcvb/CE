from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from step07_target_attack import (
    _prepare_attack_graph,
    compute_attack_curves,
    compute_centrality_ablation_attacks,
    compute_target_attack,
)


def test_static_attack_curve_includes_baseline() -> None:
    graph = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=np.float64,
    )

    curve = compute_target_attack(graph, directed=False)

    assert curve.shape == (4,)
    np.testing.assert_allclose(curve, [1.0, 0.0, 0.0, 0.0])


def test_undirected_attack_graph_is_symmetrized() -> None:
    graph = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=np.float64,
    )

    prepared = _prepare_attack_graph(graph, directed=False)

    np.testing.assert_array_equal(prepared, np.maximum(graph, graph.T))


def test_non_binary_attack_graph_is_rejected() -> None:
    graph = np.array(
        [
            [0.0, 0.5],
            [0.5, 0.0],
        ],
        dtype=np.float64,
    )

    try:
        compute_target_attack(graph)
    except ValueError as exc:
        assert "binary adjacency matrix" in str(exc)
    else:
        raise AssertionError("non-binary attack graph should be rejected")


def test_attack_controls_return_expected_strategy_curves() -> None:
    graph = np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=np.float64,
    )

    curves = compute_attack_curves(graph, random_repeats=5, seed=7)

    assert set(curves) == {"ce", "ce_adaptive", "degree", "betweenness", "random_mean", "random_std"}
    for curve in curves.values():
        assert curve.shape == (5,)
        assert np.isfinite(curve).all()
    assert curves["ce"][0] == 1.0
    assert curves["random_std"][0] == 0.0


def test_centrality_ablation_attacks_include_ce_building_blocks() -> None:
    graph = np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=np.float64,
    )

    curves = compute_centrality_ablation_attacks(graph)

    assert set(curves) == {"degree", "betweenness", "closeness", "s", "h", "ce"}
    for curve in curves.values():
        assert curve.shape == (5,)
        assert curve[0] == 1.0


if __name__ == "__main__":
    test_static_attack_curve_includes_baseline()
    test_undirected_attack_graph_is_symmetrized()
    test_non_binary_attack_graph_is_rejected()
    print("target attack smoke tests passed")
