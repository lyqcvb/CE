from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from step05_hub_detection import (
    closeness_centrality,
    compute_ce,
    compute_weighted_ce,
    critic_objective_weights,
    identify_hub_nodes,
    identify_weighted_hub_nodes,
    identify_traditional_hubs,
    topsis_closeness_scores,
)


def test_closeness_centrality_matches_harmonic_formula_on_connected_graph() -> None:
    graph = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=np.float64,
    )

    closeness = closeness_centrality(graph, directed=False)

    np.testing.assert_allclose(closeness, [0.75, 1.0, 0.75])


def test_closeness_centrality_keeps_reachable_terms_in_disconnected_graph() -> None:
    graph = np.array(
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.float64,
    )

    closeness = closeness_centrality(graph, directed=False)

    np.testing.assert_allclose(closeness, [0.5, 0.5, 0.0])


def test_closeness_centrality_uses_outgoing_paths_for_directed_graph() -> None:
    graph = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=np.float64,
    )

    closeness = closeness_centrality(graph, directed=True)

    np.testing.assert_allclose(closeness, [0.75, 0.5, 0.0])


def test_critic_objective_weights_downweight_redundant_constant_indicator() -> None:
    normalized = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.5, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    weights = critic_objective_weights(normalized)

    assert weights[2] == 0.0
    np.testing.assert_allclose(weights.sum(), 1.0)
    assert np.all(weights[:2] > 0.0)


def test_topsis_closeness_scores_rank_positive_ideal_highest() -> None:
    normalized = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    weights = np.array([0.2, 0.3, 0.5], dtype=np.float64)

    score = topsis_closeness_scores(normalized, weights)

    np.testing.assert_allclose(score, [0.0, 0.5, 1.0])


def test_topsis_closeness_scores_applies_vector_normalization_before_weighting() -> None:
    normalized = np.array(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [0.5, 1.0],
        ],
        dtype=np.float64,
    )
    weights = np.array([0.25, 0.75], dtype=np.float64)

    score = topsis_closeness_scores(normalized, weights)

    vector_norm = np.sqrt(np.sum(normalized**2, axis=0))
    vector_normalized = normalized / vector_norm
    weighted = vector_normalized * weights
    positive_ideal = weighted.max(axis=0)
    negative_ideal = weighted.min(axis=0)
    distance_positive = np.sqrt(np.sum((weighted - positive_ideal) ** 2, axis=1))
    distance_negative = np.sqrt(np.sum((weighted - negative_ideal) ** 2, axis=1))
    expected = distance_negative / (distance_positive + distance_negative)

    np.testing.assert_allclose(score, expected)


def test_compute_ce_uses_objective_scores_and_returns_finite_values() -> None:
    graph = np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=np.float64,
    )

    ce, score = compute_ce(graph, directed=False)

    assert ce.shape == (4,)
    assert score.shape == (4,)
    assert np.all(np.isfinite(ce))
    assert np.all(np.isfinite(score))
    assert np.all((score >= 0.0) & (score <= 1.0))
    assert np.all(ce <= score + 1e-12)


def test_compute_ce_uses_normalized_neighborhood_entropy() -> None:
    graph = np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0],
        ],
        dtype=np.float64,
    )

    ce, score = compute_ce(graph, directed=False)
    adjacency = graph > 0
    np.fill_diagonal(adjacency, False)

    expected_ce = np.zeros_like(ce)
    for i in range(graph.shape[0]):
        neighbors = np.where(adjacency[i])[0]
        if len(neighbors) <= 1 or score[i] <= 0:
            continue

        neighbor_scores = score[neighbors]
        neighbor_score_sum = neighbor_scores.sum()
        if neighbor_score_sum <= 0:
            continue

        probabilities = neighbor_scores / neighbor_score_sum
        valid_probabilities = probabilities[probabilities > 0]
        entropy = -np.sum(valid_probabilities * np.log2(valid_probabilities))
        expected_ce[i] = score[i] * entropy / np.log2(len(neighbors))

    np.testing.assert_allclose(ce, expected_ce)


def test_top_ratio_hub_detection_always_selects_comparable_node_count() -> None:
    graph = np.array(
        [
            [0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 0, 0],
            [0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0],
        ],
        dtype=np.float64,
    )

    is_hub, ce, threshold = identify_hub_nodes(graph, top_ratio=0.20)
    traditional = identify_traditional_hubs(graph, top_ratio=0.20)

    assert is_hub.sum() == 1
    assert threshold in ce
    assert all(mask.sum() == 1 for mask in traditional.values())


def test_weighted_ce_hub_detection_returns_finite_scores() -> None:
    weight = np.array(
        [
            [0.0, 0.9, 0.4, 0.0],
            [0.9, 0.0, 0.8, 0.3],
            [0.4, 0.8, 0.0, 0.2],
            [0.0, 0.3, 0.2, 0.0],
        ],
        dtype=np.float64,
    )

    ce, score = compute_weighted_ce(weight)
    is_hub, ce_again, threshold = identify_weighted_hub_nodes(weight, top_ratio=0.25)

    assert ce.shape == (4,)
    assert score.shape == (4,)
    assert np.isfinite(ce).all()
    assert np.isfinite(score).all()
    assert is_hub.sum() == 1
    np.testing.assert_allclose(ce_again, ce)
    assert np.isfinite(threshold)
