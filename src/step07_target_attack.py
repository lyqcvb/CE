"""
Step 07: Targeted attack simulation.
"""

from __future__ import annotations

import numpy as np

from step04_graph_metrics import global_efficiency_bin, largest_component_size
from step05_hub_detection import (
    betweenness_bin,
    compute_ce,
    compute_ce_components,
    degree_centrality,
)


def _prepare_attack_graph(graph: np.ndarray, directed: bool = False) -> np.ndarray:
    """
    Validate and clean a binary adjacency matrix for targeted attack.
    """
    graph = np.asarray(graph, dtype=np.float64).copy()
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError(f"Attack graph must be a square matrix, got {graph.shape}")

    graph = np.nan_to_num(graph, nan=0.0, posinf=0.0, neginf=0.0)
    graph[graph < 0.0] = 0.0
    np.fill_diagonal(graph, 0.0)

    if not np.all((graph == 0.0) | (graph == 1.0)):
        raise ValueError("Targeted attack expects a thresholded binary adjacency matrix")

    if not directed:
        graph = np.maximum(graph, graph.T)

    return graph


def _attack_order(ce: np.ndarray, score: np.ndarray, graph: np.ndarray, directed: bool) -> np.ndarray:
    """
    Rank nodes by descending CE with deterministic secondary keys.
    """
    degree = graph.sum(axis=1) + graph.sum(axis=0) if directed else graph.sum(axis=1)
    index = np.arange(len(ce))
    return np.lexsort((index, -degree, -score, -ce))


def _metric_order(values: np.ndarray) -> np.ndarray:
    index = np.arange(len(values))
    return np.lexsort((index, -np.asarray(values, dtype=np.float64)))


def _attack_outcome(graph: np.ndarray, outcome: str) -> float:
    if outcome == "efficiency":
        return global_efficiency_bin(graph)
    if outcome == "lcc":
        return float(largest_component_size(graph))
    raise ValueError(f"unknown attack outcome {outcome!r}")


def _curve_from_order(
    graph: np.ndarray,
    order: np.ndarray,
    eg0: float,
    include_baseline: bool,
    outcome: str = "efficiency",
) -> np.ndarray:
    """
    Outcome after isolating nodes in order, normalized by the intact value.

    outcome="efficiency" tracks binary global efficiency; outcome="lcc"
    tracks the largest connected component size, which does not share the
    closeness/efficiency term used by CE ranking.
    """
    n = graph.shape[0]
    curve_len = n + 1 if include_baseline else n
    eglob = np.zeros(curve_len, dtype=np.float64)
    if eg0 <= 0.0:
        return eglob
    if include_baseline:
        eglob[0] = 1.0

    working = graph.copy()
    offset = 1 if include_baseline else 0
    for step, node in enumerate(order[:n]):
        node = int(node)
        working[:, node] = 0.0
        working[node, :] = 0.0
        eglob[step + offset] = _attack_outcome(working, outcome) / eg0
    return eglob


def compute_target_attack(
    graph_bin: np.ndarray,
    directed: bool = False,
    eg0: float | None = None,
    include_baseline: bool = True,
    adaptive: bool = False,
) -> np.ndarray:
    """
    Simulate CE-based targeted attack and record normalized global efficiency.

    Nodes are isolated by setting all incident edges to zero while keeping the
    original node set. By default, nodes are removed according to their initial
    CE ranking. Set adaptive=True to recalculate CE after each removal.
    """
    graph = _prepare_attack_graph(graph_bin, directed=directed)
    n = graph.shape[0]

    if eg0 is None:
        eg0 = global_efficiency_bin(graph)

    curve_len = n + 1 if include_baseline else n
    eglob = np.zeros(curve_len, dtype=np.float64)

    if eg0 <= 0.0:
        return eglob

    if include_baseline:
        eglob[0] = 1.0

    working = graph.copy()
    removed = np.zeros(n, dtype=bool)
    if not adaptive:
        ce, score = compute_ce(working, directed=directed)
        order = _attack_order(ce, score, working, directed=directed)

    offset = 1 if include_baseline else 0
    for step in range(n):
        if adaptive:
            ce, score = compute_ce(working, directed=directed)
            ce[removed] = -np.inf
            score[removed] = -np.inf
            node = int(_attack_order(ce, score, working, directed=directed)[0])
        else:
            node = int(order[step])

        removed[node] = True
        working[:, node] = 0.0
        working[node, :] = 0.0
        efficiency = global_efficiency_bin(working)
        eglob[step + offset] = efficiency / eg0

    return eglob


def compute_attack_by_order(
    graph_bin: np.ndarray,
    order: np.ndarray,
    directed: bool = False,
    eg0: float | None = None,
    include_baseline: bool = True,
    outcome: str = "efficiency",
) -> np.ndarray:
    """Compute the normalized outcome after removing nodes in a supplied order."""
    graph = _prepare_attack_graph(graph_bin, directed=directed)
    if eg0 is None:
        eg0 = _attack_outcome(graph, outcome)
    return _curve_from_order(graph, np.asarray(order, dtype=int), eg0, include_baseline, outcome)


def compute_attack_curves(
    graph_bin: np.ndarray,
    directed: bool = False,
    random_repeats: int = 1000,
    seed: int = 42,
    include_baseline: bool = True,
) -> dict[str, np.ndarray]:
    """
    Compute CE, adaptive CE, degree, betweenness, and random attack controls.
    """
    graph = _prepare_attack_graph(graph_bin, directed=directed)
    n = graph.shape[0]
    eg0 = global_efficiency_bin(graph)

    ce, score = compute_ce(graph, directed=directed)
    ce_order = _attack_order(ce, score, graph, directed=directed)
    degree_order = _metric_order(degree_centrality(graph, directed=directed))
    betweenness_order = _metric_order(betweenness_bin(graph, directed=directed))

    curves = {
        "ce": _curve_from_order(graph, ce_order, eg0, include_baseline),
        "ce_adaptive": compute_target_attack(
            graph,
            directed=directed,
            eg0=eg0,
            include_baseline=include_baseline,
            adaptive=True,
        ),
        "degree": _curve_from_order(graph, degree_order, eg0, include_baseline),
        "betweenness": _curve_from_order(graph, betweenness_order, eg0, include_baseline),
    }

    rng = np.random.default_rng(seed)
    random_curves = np.zeros((max(1, random_repeats), n + 1 if include_baseline else n))
    base_order = np.arange(n)
    for idx in range(random_curves.shape[0]):
        order = rng.permutation(base_order)
        random_curves[idx] = _curve_from_order(graph, order, eg0, include_baseline)
    curves["random_mean"] = random_curves.mean(axis=0)
    curves["random_std"] = random_curves.std(axis=0, ddof=1) if random_curves.shape[0] > 1 else np.zeros(random_curves.shape[1])

    return curves


def compute_centrality_ablation_attacks(
    graph_bin: np.ndarray,
    directed: bool = False,
    include_baseline: bool = True,
) -> dict[str, np.ndarray]:
    """
    Attack curves for CE ablation centralities: Degree, BC, CC, S, H, and CE.
    """
    graph = _prepare_attack_graph(graph_bin, directed=directed)
    eg0 = global_efficiency_bin(graph)
    components = compute_ce_components(graph, directed=directed, weighted=False)
    method_names = {
        "degree": "Degree",
        "betweenness": "Betweenness",
        "closeness": "Closeness",
        "s": "S",
        "h": "H",
        "ce": "CE",
    }
    return {
        output_name: _curve_from_order(
            graph,
            _metric_order(components[component_name]),
            eg0,
            include_baseline,
        )
        for output_name, component_name in method_names.items()
    }
