"""
Step 06: Binary normalized rich-club coefficient.
"""

from __future__ import annotations

import numpy as np


def _prepare_binary_rich_graph(graph: np.ndarray, directed: bool = False) -> np.ndarray:
    """
    Validate and clean a binary adjacency matrix for rich-club analysis.
    """
    graph = np.asarray(graph, dtype=np.float64).copy()
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError(f"Rich-club graph must be a square matrix, got {graph.shape}")

    graph = np.nan_to_num(graph, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(graph, 0.0)
    if not np.all((graph == 0) | (graph == 1)):
        raise ValueError("Rich-club analysis expects a binary adjacency matrix")

    if not directed:
        graph = np.maximum(graph, graph.T)

    return graph


def _effective_klevel(degree: np.ndarray, klevel: int | None) -> int:
    max_k = int(np.nanmax(degree)) - 1 if degree.size else 0
    if klevel is None:
        return max(0, max_k)
    return max(0, min(int(klevel), max_k))


def _nanmean_axis0(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    counts = valid.sum(axis=0)
    sums = np.where(valid, values, 0.0).sum(axis=0)
    out = np.full(values.shape[1], np.nan, dtype=np.float64)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def rich_club_bu(graph: np.ndarray, klevel: int | None = None) -> np.ndarray:
    """
    Binary undirected rich-club coefficient.
    """
    graph = _prepare_binary_rich_graph(graph, directed=False)
    degree = graph.sum(axis=1)
    klevel = _effective_klevel(degree, klevel)
    rich_club = np.full(klevel, np.nan, dtype=np.float64)
    for k in range(1, klevel + 1):
        nodes_k = np.where(degree > k)[0]
        nk = len(nodes_k)
        if nk < 2:
            continue
        subgraph = graph[np.ix_(nodes_k, nodes_k)]
        edges_k = subgraph.sum() / 2.0
        max_edges_k = nk * (nk - 1) / 2.0
        if max_edges_k > 0:
            rich_club[k - 1] = edges_k / max_edges_k
    return rich_club


def rich_club_bd(graph: np.ndarray, klevel: int | None = None) -> np.ndarray:
    """
    Binary directed rich-club coefficient.
    """
    graph = _prepare_binary_rich_graph(graph, directed=True)
    degree = graph.sum(axis=1) + graph.sum(axis=0)
    klevel = _effective_klevel(degree, klevel)
    rich_club = np.full(klevel, np.nan, dtype=np.float64)
    for k in range(1, klevel + 1):
        nodes_k = np.where(degree > k)[0]
        nk = len(nodes_k)
        if nk < 2:
            continue
        subgraph = graph[np.ix_(nodes_k, nodes_k)]
        edges_k = subgraph.sum()
        max_edges_k = nk * (nk - 1)
        if max_edges_k > 0:
            rich_club[k - 1] = edges_k / max_edges_k
    return rich_club


def _randmio_und(graph: np.ndarray, itr: int, rng: np.random.Generator) -> np.ndarray:
    """
    Randomly rewire an undirected binary graph while preserving degree sequence.
    """
    randomized = graph.copy()
    rows, cols = np.where(np.triu(randomized, 1))
    edges = list(zip(rows.tolist(), cols.tolist()))
    edge_count = len(edges)
    if edge_count < 2:
        return randomized

    max_attempts = itr * edge_count
    for _ in range(max_attempts):
        idx = rng.integers(0, edge_count, size=2)
        if idx[0] == idx[1]:
            continue
        a, b = edges[idx[0]]
        c, d = edges[idx[1]]
        if rng.random() > 0.5:
            i1, j1, i2, j2 = a, d, c, b
        else:
            i1, j1, i2, j2 = a, c, b, d

        if i1 == j1 or i2 == j2:
            continue
        if i1 > j1:
            i1, j1 = j1, i1
        if i2 > j2:
            i2, j2 = j2, i2

        if i1 == i2 and j1 == j2:
            continue

        if randomized[i1, j1] == 0 and randomized[i2, j2] == 0:
            randomized[a, b] = randomized[b, a] = 0
            randomized[c, d] = randomized[d, c] = 0
            randomized[i1, j1] = randomized[j1, i1] = 1
            randomized[i2, j2] = randomized[j2, i2] = 1
            edges[idx[0]] = (i1, j1)
            edges[idx[1]] = (i2, j2)
    return randomized


def _randmio_dir(graph: np.ndarray, itr: int, rng: np.random.Generator) -> np.ndarray:
    """
    Randomly rewire a directed binary graph while preserving degree sequence.
    """
    randomized = graph.copy()
    rows, cols = np.where(randomized)
    edges = list(zip(rows.tolist(), cols.tolist()))
    edge_count = len(edges)
    if edge_count < 2:
        return randomized

    max_attempts = itr * edge_count
    for _ in range(max_attempts):
        edge_1 = int(rng.integers(0, edge_count))
        edge_2 = int(rng.integers(0, edge_count))
        if edge_1 == edge_2:
            continue
        a, b = edges[edge_1]
        c, d = edges[edge_2]
        if a == c or b == d or a == d or c == b:
            continue
        if randomized[a, d] == 0 and randomized[c, b] == 0:
            randomized[a, b] = 0
            randomized[c, d] = 0
            randomized[a, d] = 1
            randomized[c, b] = 1
            edges[edge_1] = (a, d)
            edges[edge_2] = (c, b)
    return randomized


def compute_binary_normalized_rich_club(
    graph_bin: np.ndarray,
    directed: bool = False,
    klevel: int | None = None,
    n_rand: int = 1000,
    seed: int = 42,
    rewire_iter: int = 10,
) -> np.ndarray:
    """
    Normalized rich-club coefficient on a binary thresholded graph.
    """
    graph_bin = _prepare_binary_rich_graph(graph_bin, directed=directed)

    if directed:
        rich_club = rich_club_bd(graph_bin, klevel)
    else:
        rich_club = rich_club_bu(graph_bin, klevel)

    klevel = len(rich_club)
    if klevel == 0:
        return rich_club

    rng = np.random.default_rng(seed)
    random_rich_club = np.full((n_rand, klevel), np.nan, dtype=np.float64)
    for idx in range(n_rand):
        if directed:
            randomized = _randmio_dir(graph_bin, rewire_iter, rng)
            random_rich_club[idx] = rich_club_bd(randomized, klevel)
        else:
            randomized = _randmio_und(graph_bin, rewire_iter, rng)
            random_rich_club[idx] = rich_club_bu(randomized, klevel)

    random_mean = _nanmean_axis0(random_rich_club)
    normalized = np.full_like(rich_club, np.nan, dtype=np.float64)
    valid = np.isfinite(rich_club) & np.isfinite(random_mean) & (random_mean > 0)
    normalized[valid] = rich_club[valid] / random_mean[valid]
    return normalized


def weighted_rich_club(
    weight: np.ndarray,
    directed: bool = False,
    klevel: int | None = None,
) -> np.ndarray:
    """
    Weighted rich-club coefficient using degree-defined rich nodes.

    The coefficient is the total weight among rich-club nodes divided by the sum
    of the strongest E_k weights in the full graph, where E_k is the number of
    edges among nodes whose binary degree is greater than k.
    """
    _, coefficient = weighted_rich_club_observed(weight, directed=directed, klevel=klevel)
    return coefficient


def _prepare_weighted_rich_graph(
    weight: np.ndarray,
    directed: bool = False,
) -> np.ndarray:
    graph = np.asarray(weight, dtype=np.float64).copy()
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError(f"weight must be a square matrix, got {graph.shape}")

    graph = np.nan_to_num(graph, nan=0.0, posinf=0.0, neginf=0.0)
    graph[graph < 0.0] = 0.0
    np.fill_diagonal(graph, 0.0)
    if not directed:
        graph = np.maximum(graph, graph.T)
    return graph


def weighted_rich_club_observed(
    weight: np.ndarray,
    directed: bool = False,
    klevel: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return raw rich-club weight sums and rank-normalized weighted rich-club.
    """
    graph = _prepare_weighted_rich_graph(weight, directed=directed)
    if not directed:
        binary = (graph > 0.0).astype(np.float64)
        degree = binary.sum(axis=1)
        rows, cols = np.triu_indices(graph.shape[0], k=1)
    else:
        binary = (graph > 0.0).astype(np.float64)
        degree = binary.sum(axis=1) + binary.sum(axis=0)
        rows, cols = np.where(~np.eye(graph.shape[0], dtype=bool))

    klevel = _effective_klevel(degree, klevel)
    observed_weight = np.full(klevel, np.nan, dtype=np.float64)
    rank_normalized = np.full(klevel, np.nan, dtype=np.float64)
    all_weights = np.sort(graph[rows, cols][graph[rows, cols] > 0.0])[::-1]
    if all_weights.size == 0:
        return observed_weight, rank_normalized

    for k in range(1, klevel + 1):
        nodes_k = np.where(degree > k)[0]
        if len(nodes_k) < 2:
            continue
        subgraph = graph[np.ix_(nodes_k, nodes_k)]
        if not directed:
            rich_weights = subgraph[np.triu_indices(len(nodes_k), k=1)]
        else:
            rich_weights = subgraph[~np.eye(len(nodes_k), dtype=bool)]
        rich_weights = rich_weights[rich_weights > 0.0]
        edge_count = rich_weights.size
        if edge_count == 0:
            continue
        denominator = all_weights[:edge_count].sum()
        observed_weight[k - 1] = rich_weights.sum()
        if denominator > 0.0:
            rank_normalized[k - 1] = observed_weight[k - 1] / denominator

    return observed_weight, rank_normalized


def _random_weighted_graph(
    weight: np.ndarray,
    directed: bool,
    rewire_iter: int,
    rng: np.random.Generator,
) -> np.ndarray:
    graph = _prepare_weighted_rich_graph(weight, directed=directed)
    binary = (graph > 0.0).astype(np.float64)
    if directed:
        randomized_binary = _randmio_dir(binary, rewire_iter, rng)
        source_rows, source_cols = np.where(binary > 0.0)
        target_rows, target_cols = np.where(randomized_binary > 0.0)
    else:
        randomized_binary = _randmio_und(binary, rewire_iter, rng)
        source_rows, source_cols = np.triu_indices(graph.shape[0], k=1)
        source_mask = binary[source_rows, source_cols] > 0.0
        source_rows = source_rows[source_mask]
        source_cols = source_cols[source_mask]
        target_rows, target_cols = np.where(np.triu(randomized_binary, 1) > 0.0)

    weights = graph[source_rows, source_cols]
    weights = weights[weights > 0.0]
    if weights.size == 0:
        return np.zeros_like(graph)

    assigned = rng.permutation(weights)
    if assigned.size != target_rows.size:
        assigned = np.resize(assigned, target_rows.size)

    randomized = np.zeros_like(graph)
    randomized[target_rows, target_cols] = assigned
    if not directed:
        randomized[target_cols, target_rows] = assigned
    return randomized


def compute_weighted_normalized_rich_club(
    weight: np.ndarray,
    directed: bool = False,
    klevel: int | None = None,
    n_rand: int = 1000,
    seed: int = 42,
    rewire_iter: int = 10,
) -> dict[str, np.ndarray]:
    """
    Weighted rich-club with random-network normalization and empirical p-values.
    """
    observed_weight, rank_normalized = weighted_rich_club_observed(
        weight,
        directed=directed,
        klevel=klevel,
    )
    klevel = len(rank_normalized)
    if klevel == 0:
        empty = np.array([], dtype=np.float64)
        return {
            "Phi_w_observed": empty,
            "Phi_w_ranknorm": empty,
            "Phi_w_random_mean": empty,
            "Phi_w_random_std": empty,
            "Phi_w_norm": empty,
            "Phi_w_empirical_p": empty,
        }

    rng = np.random.default_rng(seed)
    random_coefficients = np.full((n_rand, klevel), np.nan, dtype=np.float64)
    for idx in range(n_rand):
        randomized = _random_weighted_graph(weight, directed, rewire_iter, rng)
        random_coefficients[idx] = weighted_rich_club(
            randomized,
            directed=directed,
            klevel=klevel,
        )

    random_mean = _nanmean_axis0(random_coefficients)
    random_std = np.full(klevel, np.nan, dtype=np.float64)
    valid_counts = np.isfinite(random_coefficients).sum(axis=0)
    for idx in range(klevel):
        valid = random_coefficients[:, idx][np.isfinite(random_coefficients[:, idx])]
        if valid.size > 1:
            random_std[idx] = float(np.std(valid, ddof=1))
        elif valid.size == 1:
            random_std[idx] = 0.0

    normalized = np.full(klevel, np.nan, dtype=np.float64)
    valid_norm = np.isfinite(rank_normalized) & np.isfinite(random_mean) & (random_mean > 0.0)
    normalized[valid_norm] = rank_normalized[valid_norm] / random_mean[valid_norm]

    empirical_p = np.full(klevel, np.nan, dtype=np.float64)
    for idx in range(klevel):
        observed = rank_normalized[idx]
        random_values = random_coefficients[:, idx]
        random_values = random_values[np.isfinite(random_values)]
        if np.isfinite(observed) and random_values.size:
            empirical_p[idx] = (np.sum(random_values >= observed) + 1.0) / (
                random_values.size + 1.0
            )

    return {
        "Phi_w_observed": observed_weight,
        "Phi_w_ranknorm": rank_normalized,
        "Phi_w_random_mean": random_mean,
        "Phi_w_random_std": random_std,
        "Phi_w_norm": normalized,
        "Phi_w_empirical_p": empirical_p,
        "Phi_w_random_valid_n": valid_counts.astype(np.float64),
    }
