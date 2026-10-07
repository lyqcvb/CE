"""
Null models for weighted complete graphs and fixed-density binary graphs.

Degree-preserving edge swaps change nothing on a complete weighted graph, so
weighted metrics are normalized against:
- strength_preserving_null: Rubinov & Sporns (2011) weight reassignment
  (BCT null_model_und_sign for non-negative weights on a fixed topology).
  The weight multiset is preserved exactly and the strength sequence
  approximately, so the null tracks the mean-connectivity level.
- weight_permutation_null: random permutation of edge weights (weight
  distribution only).
Binary fixed-density graphs use Maslov-Sneppen degree-preserving rewiring.
"""

from __future__ import annotations

import numpy as np


def _upper_edges(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(weight.shape[0], k=1)


def weight_permutation_null(weight: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle undirected edge weights among node pairs."""
    w = np.asarray(weight, dtype=np.float64)
    rows, cols = _upper_edges(w)
    values = rng.permutation(w[rows, cols])
    out = np.zeros_like(w)
    out[rows, cols] = values
    out[cols, rows] = values
    return out


def strength_preserving_null(weight: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Reassign weights on the existing (positive) edges so strengths are approximately kept.

    At each step a random rank r is drawn among the m remaining edges; the
    edge with the r-th smallest expected weight (s_i * s_j from the remaining
    strengths) receives the r-th smallest remaining weight.
    """
    w = np.asarray(weight, dtype=np.float64).copy()
    w = np.maximum(w, w.T)
    np.fill_diagonal(w, 0.0)
    w[w < 0.0] = 0.0
    rows, cols = _upper_edges(w)
    present = w[rows, cols] > 0.0
    rows, cols = rows[present], cols[present]
    weights = np.sort(w[rows, cols])
    strength = w.sum(axis=1)

    out = np.zeros_like(w)
    remaining = np.ones(rows.size, dtype=bool)
    remaining_weights = list(weights)
    for m in range(rows.size, 0, -1):
        idx = np.flatnonzero(remaining)
        expected = strength[rows[idx]] * strength[cols[idx]]
        order = idx[np.argsort(expected, kind="stable")]
        r = int(rng.integers(m))
        edge = order[r]
        value = remaining_weights.pop(r)
        i, j = rows[edge], cols[edge]
        out[i, j] = out[j, i] = value
        strength[i] = max(strength[i] - value, 0.0)
        strength[j] = max(strength[j] - value, 0.0)
        remaining[edge] = False
    return out


def degree_preserving_rewire(graph_bin: np.ndarray, rng: np.random.Generator, swaps_per_edge: int = 10) -> np.ndarray:
    """Maslov-Sneppen double-edge swaps on an undirected binary graph."""
    g = (np.asarray(graph_bin) > 0).astype(np.int8)
    g = np.maximum(g, g.T)
    np.fill_diagonal(g, 0)
    rows, cols = np.nonzero(np.triu(g, 1))
    edges = np.column_stack([rows, cols])
    n_edges = edges.shape[0]
    if n_edges < 2:
        return g.astype(np.float64)
    attempts = swaps_per_edge * n_edges
    for _ in range(attempts):
        e1, e2 = rng.integers(n_edges, size=2)
        if e1 == e2:
            continue
        a, b = edges[e1]
        c, d = edges[e2]
        if rng.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4 or g[a, d] or g[c, b]:
            continue
        g[a, b] = g[b, a] = 0
        g[c, d] = g[d, c] = 0
        g[a, d] = g[d, a] = 1
        g[c, b] = g[b, c] = 1
        edges[e1] = (a, d)
        edges[e2] = (c, b)
    return g.astype(np.float64)


def null_normalized(metric, graph: np.ndarray, null_fn, repeats: int, seed: int) -> tuple[float, float]:
    """Return (observed / mean null, mean null) for a scalar graph metric."""
    rng = np.random.default_rng(seed)
    observed = float(metric(graph))
    null_values = np.array([metric(null_fn(graph, rng)) for _ in range(repeats)], dtype=np.float64)
    mean_null = float(np.mean(null_values))
    ratio = observed / mean_null if mean_null > 0 else np.nan
    return ratio, mean_null
