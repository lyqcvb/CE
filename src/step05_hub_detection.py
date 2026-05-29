"""
Step 05: Hub-node detection with centrality entropy.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from step04_graph_metrics import _bfs_distance


def _prepare_binary_ce_graph(graph: np.ndarray, directed: bool = False) -> np.ndarray:
    """
    Normalize CE input to a clean thresholded binary adjacency matrix.

    External callers should still apply the intended threshold before calling
    CE. This sanitizer prevents self-connections, NaN/inf, negative values, or
    residual magnitudes from leaking into centrality calculations.
    """
    adjacency = np.asarray(graph, dtype=np.float64).copy()
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"CE graph must be a square matrix, got {adjacency.shape}")

    adjacency = np.nan_to_num(adjacency, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(adjacency, 0.0)
    adjacency[adjacency < 0.0] = 0.0
    adjacency = (adjacency > 0.0).astype(np.float64)

    if not directed:
        adjacency = np.maximum(adjacency, adjacency.T)

    return adjacency


def _prepare_weighted_ce_graph(weight: np.ndarray, directed: bool = False) -> np.ndarray:
    """Normalize weighted CE input to a clean non-negative adjacency matrix."""
    graph = np.asarray(weight, dtype=np.float64).copy()
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError(f"Weighted CE graph must be a square matrix, got {graph.shape}")

    graph = np.nan_to_num(graph, nan=0.0, posinf=0.0, neginf=0.0)
    graph[graph < 0.0] = 0.0
    np.fill_diagonal(graph, 0.0)
    if not directed:
        graph = np.maximum(graph, graph.T)
    return graph


def degree_centrality(graph: np.ndarray, directed: bool = False) -> np.ndarray:
    """
    Degree centrality used by CE on a thresholded binary network.
    """
    graph = _prepare_binary_ce_graph(graph, directed=directed)
    if directed:
        return graph.sum(axis=1) + graph.sum(axis=0)
    return graph.sum(axis=1)


def betweenness_bin(graph: np.ndarray, directed: bool = False) -> np.ndarray:
    """
    Betweenness centrality used by CE on a thresholded binary network.

    Directed mode follows outgoing edge direction; LOFC/HOFC use undirected graphs.
    """
    graph = _prepare_binary_ce_graph(graph, directed=directed)
    n = graph.shape[0]
    graph_x = nx.from_numpy_array(
        graph,
        create_using=nx.DiGraph() if directed else nx.Graph(),
    )
    bc_dict = nx.betweenness_centrality(graph_x, normalized=False, weight=None)
    scale = 1.0 if directed else 2.0
    return np.array([bc_dict[i] * scale for i in range(n)])


def closeness_centrality(graph: np.ndarray, directed: bool = False) -> np.ndarray:
    """
    Harmonic closeness/nodal efficiency used by CE on a binary network.

    For directed graphs, shortest paths follow outgoing edge direction.
    Unreachable nodes contribute 0 instead of forcing the whole node score to 0.
    """
    graph = _prepare_binary_ce_graph(graph, directed=directed)
    distance = _bfs_distance(graph)
    n = graph.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.float64)

    finite_off_diagonal = np.isfinite(distance) & (distance > 0)
    inverse_distance = np.zeros_like(distance, dtype=np.float64)
    inverse_distance[finite_off_diagonal] = 1.0 / distance[finite_off_diagonal]

    return inverse_distance.sum(axis=1) / (n - 1)


def strength_centrality(weight: np.ndarray, directed: bool = False) -> np.ndarray:
    """Weighted degree/strength centrality."""
    graph = _prepare_weighted_ce_graph(weight, directed=directed)
    if directed:
        return graph.sum(axis=1) + graph.sum(axis=0)
    return graph.sum(axis=1)


def betweenness_weighted(weight: np.ndarray, directed: bool = False) -> np.ndarray:
    """Weighted betweenness centrality using edge length = 1 / weight."""
    graph = _prepare_weighted_ce_graph(weight, directed=directed)
    n = graph.shape[0]
    graph_x = nx.from_numpy_array(
        graph,
        create_using=nx.DiGraph() if directed else nx.Graph(),
    )
    length = {
        (u, v): 1.0 / data["weight"]
        for u, v, data in graph_x.edges(data=True)
        if data.get("weight", 0.0) > 0.0
    }
    nx.set_edge_attributes(graph_x, length, "length")
    bc_dict = nx.betweenness_centrality(graph_x, normalized=False, weight="length")
    scale = 1.0 if directed else 2.0
    return np.array([bc_dict[i] * scale for i in range(n)], dtype=np.float64)


def closeness_weighted(weight: np.ndarray, directed: bool = False) -> np.ndarray:
    """Weighted harmonic closeness with edge length = 1 / weight."""
    graph = _prepare_weighted_ce_graph(weight, directed=directed)
    n = graph.shape[0]
    graph_x = nx.from_numpy_array(
        graph,
        create_using=nx.DiGraph() if directed else nx.Graph(),
    )
    length = {
        (u, v): 1.0 / data["weight"]
        for u, v, data in graph_x.edges(data=True)
        if data.get("weight", 0.0) > 0.0
    }
    nx.set_edge_attributes(graph_x, length, "length")

    closeness = np.zeros(n, dtype=np.float64)
    for source, distances in nx.all_pairs_dijkstra_path_length(graph_x, weight="length"):
        inv_sum = sum(1.0 / distance for target, distance in distances.items() if target != source and distance > 0.0)
        closeness[int(source)] = inv_sum / (n - 1) if n > 1 else 0.0
    return closeness


def critic_objective_weights(normalized: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Calculate fully objective indicator weights with the CRITIC method.

    CRITIC combines each indicator's contrast intensity (standard deviation)
    and its conflict with other indicators (1 - correlation). Constant or
    otherwise non-informative indicators receive zero information weight unless
    all indicators are non-informative, in which case the function falls back to
    equal weights.
    """
    normalized = np.asarray(normalized, dtype=np.float64)
    if normalized.ndim != 2:
        raise ValueError("normalized must be a 2D decision matrix.")

    n_indicators = normalized.shape[1]
    if n_indicators == 0:
        raise ValueError("normalized must contain at least one indicator.")

    if normalized.shape[0] <= 1:
        return np.full(n_indicators, 1.0 / n_indicators, dtype=np.float64)

    centered = normalized - np.nanmean(normalized, axis=0)
    centered = np.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.sqrt(np.mean(centered**2, axis=0))

    denominator = np.sqrt(np.sum(centered**2, axis=0))
    numerator = centered.T @ centered
    corr = np.zeros((n_indicators, n_indicators), dtype=np.float64)
    valid_denominator = denominator > eps
    valid_pairs = np.outer(valid_denominator, valid_denominator)
    corr[valid_pairs] = numerator[valid_pairs] / np.outer(denominator, denominator)[valid_pairs]
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    information = std * np.sum(1.0 - corr, axis=1)
    information = np.nan_to_num(information, nan=0.0, posinf=0.0, neginf=0.0)
    information[information < eps] = 0.0

    information_sum = information.sum()
    if information_sum <= eps or not np.isfinite(information_sum):
        return np.full(n_indicators, 1.0 / n_indicators, dtype=np.float64)

    return information / information_sum


def topsis_closeness_scores(
    normalized: np.ndarray,
    weights: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Calculate node scores as TOPSIS relative closeness to the positive ideal.

    All CE indicators are benefit-type after min-max normalization. TOPSIS then
    applies column-wise vector normalization before weighting. The returned
    score is D- / (D+ + D-), where D+ is the distance to the positive ideal and
    D- is the distance to the negative ideal.
    """
    normalized = np.asarray(normalized, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if normalized.ndim != 2:
        raise ValueError("normalized must be a 2D decision matrix.")
    if weights.ndim != 1 or weights.shape[0] != normalized.shape[1]:
        raise ValueError("weights must match the number of decision-matrix columns.")

    weight_sum = weights.sum()
    if weight_sum <= eps or not np.isfinite(weight_sum):
        weights = np.full(normalized.shape[1], 1.0 / normalized.shape[1], dtype=np.float64)
    else:
        weights = weights / weight_sum

    vector_norm = np.sqrt(np.nansum(normalized**2, axis=0))
    vector_normalized = np.zeros_like(normalized, dtype=np.float64)
    valid_columns = vector_norm > eps
    vector_normalized[:, valid_columns] = (
        normalized[:, valid_columns] / vector_norm[valid_columns]
    )

    weighted = vector_normalized * weights
    positive_ideal = np.nanmax(weighted, axis=0)
    negative_ideal = np.nanmin(weighted, axis=0)
    distance_positive = np.sqrt(np.nansum((weighted - positive_ideal) ** 2, axis=1))
    distance_negative = np.sqrt(np.nansum((weighted - negative_ideal) ** 2, axis=1))
    denominator = distance_positive + distance_negative

    score = np.zeros(normalized.shape[0], dtype=np.float64)
    valid = denominator > eps
    score[valid] = distance_negative[valid] / denominator[valid]
    return score


def _minmax_normalize_decision_matrix(decision_matrix: np.ndarray) -> np.ndarray:
    x_min = decision_matrix.min(axis=0)
    x_max = decision_matrix.max(axis=0)
    x_range = x_max - x_min
    x_range[x_range == 0] = 1.0
    return (decision_matrix - x_min) / x_range


def _centrality_entropy_from_score(
    graph: np.ndarray,
    score: np.ndarray,
    directed: bool = False,
) -> np.ndarray:
    n = graph.shape[0]
    entropy = np.zeros(n, dtype=np.float64)
    adjacency = (graph > 0) | (graph.T > 0) if directed else (graph > 0)
    np.fill_diagonal(adjacency, False)

    for i in range(n):
        neighbors = np.where(adjacency[i])[0]
        neighbor_count = len(neighbors)
        if neighbor_count <= 1:
            continue

        neighbor_scores = score[neighbors]
        neighbor_score_sum = neighbor_scores.sum()
        if neighbor_score_sum <= 0:
            continue

        probabilities = neighbor_scores / neighbor_score_sum
        valid_probabilities = probabilities[probabilities > 0]
        raw_entropy = -np.sum(valid_probabilities * np.log2(valid_probabilities))
        entropy[i] = raw_entropy / np.log2(neighbor_count)

    return entropy


def compute_ce_components(
    graph: np.ndarray,
    directed: bool = False,
    weighted: bool = False,
) -> dict[str, np.ndarray]:
    """
    Return CE building blocks for ablation analysis.

    Keys are Degree/Betweenness/Closeness, S, H, and CE. For weighted=True,
    Degree is weighted strength and path metrics use edge length = 1 / weight.
    """
    if weighted:
        prepared = _prepare_weighted_ce_graph(graph, directed=directed)
        degree = strength_centrality(prepared, directed=directed)
        betweenness = betweenness_weighted(prepared, directed=directed)
        closeness = closeness_weighted(prepared, directed=directed)
    else:
        prepared = _prepare_binary_ce_graph(graph, directed=directed)
        degree = degree_centrality(prepared, directed=directed)
        betweenness = betweenness_bin(prepared, directed=directed)
        closeness = closeness_centrality(prepared, directed=directed)

    decision_matrix = np.column_stack((degree, betweenness, closeness))
    normalized = _minmax_normalize_decision_matrix(decision_matrix)
    objective_weights = critic_objective_weights(normalized, eps=1e-12)
    score = topsis_closeness_scores(normalized, objective_weights, eps=1e-12)
    entropy = _centrality_entropy_from_score(prepared, score, directed=directed)
    ce = score * entropy

    return {
        "Degree": degree,
        "Betweenness": betweenness,
        "Closeness": closeness,
        "S": score,
        "H": entropy,
        "CE": ce,
    }


def compute_ce(graph: np.ndarray, directed: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute node centrality entropy (CE) on a thresholded binary network.

    For directed graphs, degree/BC/CC use directed paths. The local entropy
    neighborhood is the union of out-neighbors and in-neighbors, and the
    neighborhood entropy is normalized by log2(k_i).
    """
    components = compute_ce_components(graph, directed=directed, weighted=False)
    return components["CE"], components["S"]


def compute_weighted_ce(weight: np.ndarray, directed: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Compute weighted CE and weighted TOPSIS score."""
    components = compute_ce_components(weight, directed=directed, weighted=True)
    return components["CE"], components["S"]


def identify_hub_nodes(
    graph: np.ndarray,
    directed: bool = False,
    threshold_method: str = "top_ratio",
    top_ratio: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Identify hub nodes using CE.

    threshold_method:
    - "mean_std": CE > mean(CE) + std(CE).
    - "top_ratio": top-ranked CE nodes by proportion.
    """
    ce, _ = compute_ce(graph, directed=directed)
    n = len(ce)

    if threshold_method == "mean_std":
        threshold = float(ce.mean() + ce.std())
        is_hub = ce > threshold
    elif threshold_method == "top_ratio":
        if not 0 < top_ratio <= 1:
            raise ValueError("top_ratio must be in (0, 1].")
        k = max(1, int(np.ceil(n * top_ratio)))
        order = np.argsort(ce)[::-1]
        is_hub = np.zeros(n, dtype=bool)
        is_hub[order[:k]] = True
        threshold = float(ce[order[k - 1]])
    else:
        raise ValueError("threshold_method must be 'mean_std' or 'top_ratio'.")

    return is_hub, ce, threshold


def top_ratio_mask(values: np.ndarray, top_ratio: float = 0.20) -> np.ndarray:
    """Return a deterministic mask for the top-ranked proportion of nodes."""
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError("values must be a 1D array.")
    if not 0 < top_ratio <= 1:
        raise ValueError("top_ratio must be in (0, 1].")
    n = len(scores)
    if n == 0:
        return np.zeros(0, dtype=bool)
    k = max(1, int(np.ceil(n * top_ratio)))
    index = np.arange(n)
    order = np.lexsort((index, -scores))
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def identify_traditional_hubs(
    graph: np.ndarray,
    directed: bool = False,
    top_ratio: float = 0.20,
) -> dict[str, np.ndarray]:
    """
    Identify traditional centrality hubs for comparison with CE hubs.
    """
    graph = _prepare_binary_ce_graph(graph, directed=directed)
    degree = degree_centrality(graph, directed=directed)
    betweenness = betweenness_bin(graph, directed=directed)
    closeness = closeness_centrality(graph, directed=directed)
    return {
        "Degree": top_ratio_mask(degree, top_ratio=top_ratio),
        "Betweenness": top_ratio_mask(betweenness, top_ratio=top_ratio),
        "Closeness": top_ratio_mask(closeness, top_ratio=top_ratio),
    }


def identify_weighted_hub_nodes(
    weight: np.ndarray,
    directed: bool = False,
    top_ratio: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Identify weighted CE hubs by top-ratio."""
    ce, _ = compute_weighted_ce(weight, directed=directed)
    is_hub = top_ratio_mask(ce, top_ratio=top_ratio)
    threshold = float(np.min(ce[is_hub])) if is_hub.any() else float("nan")
    return is_hub, ce, threshold
