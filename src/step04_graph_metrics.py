"""
Step 04: Graph-theory metrics.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import shortest_path


def _bfs_distance(graph: np.ndarray) -> np.ndarray:
    """
    All-source shortest-path distance for binary graphs (unweighted hop count).

    Level-synchronous BFS from all sources at once via boolean matrix products;
    row s follows outgoing edges graph[u, v] > 0. Much faster than per-call
    scipy validation for the small (16-19 node) graphs used here.
    """
    adjacency = (np.asarray(graph) > 0).astype(np.float64)
    n = adjacency.shape[0]
    distance = np.full((n, n), np.inf)
    np.fill_diagonal(distance, 0.0)
    reached = np.eye(n, dtype=bool)
    frontier = reached.astype(np.float64)
    step = 0
    while True:
        step += 1
        nxt = ((frontier @ adjacency) > 0) & ~reached
        if not nxt.any():
            break
        distance[nxt] = step
        reached |= nxt
        frontier = nxt.astype(np.float64)
    return distance


def global_efficiency_bin(graph: np.ndarray) -> float:
    """
    Binary global efficiency on a thresholded graph.
    """
    graph = (np.asarray(graph) > 0).astype(np.float64)
    np.fill_diagonal(graph, 0.0)
    n = graph.shape[0]
    if n < 2:
        return 0.0
    distance = _bfs_distance(graph)
    with np.errstate(divide="ignore"):
        inv_distance = np.where(
            np.isinf(distance),
            0.0,
            1.0 / np.where(distance == 0, 1.0, distance),
        )
    np.fill_diagonal(inv_distance, 0.0)
    return float(inv_distance.sum() / (n * (n - 1)))


def _dijkstra_distance(weight: np.ndarray) -> np.ndarray:
    """
    All-source shortest-path distance using edge length = 1 / weight.
    """
    weight = np.asarray(weight, dtype=np.float64)
    length = np.zeros_like(weight)
    positive = weight > 0.0
    length[positive] = 1.0 / weight[positive]
    return shortest_path(length, method="D", directed=True)


def global_efficiency_weighted(weight: np.ndarray, directed: bool = False) -> float:
    """
    Weighted global efficiency with shortest-path lengths defined as 1 / weight.
    """
    graph = np.asarray(weight, dtype=np.float64).copy()
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError(f"weight must be a square matrix, got {graph.shape}")

    graph = np.nan_to_num(graph, nan=0.0, posinf=0.0, neginf=0.0)
    graph[graph < 0.0] = 0.0
    np.fill_diagonal(graph, 0.0)
    if not directed:
        graph = np.maximum(graph, graph.T)

    n = graph.shape[0]
    if n < 2:
        return 0.0

    distance = _dijkstra_distance(graph)
    inverse_distance = np.zeros_like(distance, dtype=np.float64)
    valid = np.isfinite(distance) & (distance > 0.0)
    inverse_distance[valid] = 1.0 / distance[valid]
    np.fill_diagonal(inverse_distance, 0.0)
    return float(inverse_distance.sum() / (n * (n - 1)))


def clustering_coef_bin_und(graph_bin: np.ndarray) -> np.ndarray:
    """
    Binary undirected clustering coefficient on a thresholded graph.
    """
    graph = (np.asarray(graph_bin) > 0).astype(np.float64)
    graph = np.maximum(graph, graph.T)
    np.fill_diagonal(graph, 0.0)

    degree = graph.sum(axis=1)
    triangles_twice = np.diag(graph @ graph @ graph)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(degree >= 2, triangles_twice / (degree * (degree - 1)), 0.0)


def clustering_coef_bin_dir(graph_bin: np.ndarray) -> np.ndarray:
    """
    Binary directed clustering coefficient on a thresholded graph.
    """
    graph = (np.asarray(graph_bin) > 0).astype(np.float64)
    np.fill_diagonal(graph, 0.0)

    total_degree = (graph + graph.T).sum(axis=1)
    reciprocal_degree = np.diag(graph @ graph)
    cycle_count = np.diag((graph + graph.T) @ (graph + graph.T) @ (graph + graph.T))
    denom = 2.0 * (total_degree * (total_degree - 1.0) - 2.0 * reciprocal_degree)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, cycle_count / denom, 0.0)


def _normalize_weights(weight: np.ndarray, directed: bool = False) -> np.ndarray:
    graph = np.asarray(weight, dtype=np.float64).copy()
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError(f"weight must be a square matrix, got {graph.shape}")
    graph = np.nan_to_num(graph, nan=0.0, posinf=0.0, neginf=0.0)
    graph[graph < 0.0] = 0.0
    np.fill_diagonal(graph, 0.0)
    if not directed:
        graph = np.maximum(graph, graph.T)
    max_weight = float(np.nanmax(graph)) if graph.size else 0.0
    if max_weight > 0.0:
        graph = graph / max_weight
    return graph


def clustering_coef_wu(weight: np.ndarray) -> np.ndarray:
    """
    Weighted undirected clustering coefficient using the Onnela/BCT formula.
    """
    graph = _normalize_weights(weight, directed=False)
    binary = (graph > 0.0).astype(np.float64)
    degree = binary.sum(axis=1)
    weight_13 = np.power(graph, 1.0 / 3.0)
    triangles_twice = np.diag(weight_13 @ weight_13 @ weight_13)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(degree >= 2.0, triangles_twice / (degree * (degree - 1.0)), 0.0)


def clustering_coef_wd(weight: np.ndarray) -> np.ndarray:
    """
    Weighted directed clustering coefficient using a Fagiolo/BCT-style formula.
    """
    graph = _normalize_weights(weight, directed=True)
    binary = (graph > 0.0).astype(np.float64)
    total_degree = (binary + binary.T).sum(axis=1)
    reciprocal_degree = np.diag(binary @ binary)
    weight_13 = np.power(graph, 1.0 / 3.0)
    cycle_count = np.diag(
        (weight_13 + weight_13.T) @ (weight_13 + weight_13.T) @ (weight_13 + weight_13.T)
    )
    denom = 2.0 * (total_degree * (total_degree - 1.0) - 2.0 * reciprocal_degree)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0.0, cycle_count / denom, 0.0)


def graph_qc_metrics(graph_bin: np.ndarray, directed: bool = False) -> dict[str, float]:
    """
    Density, average degree, edge count, and component diagnostics for a binary graph.
    """
    graph = (np.asarray(graph_bin) > 0.0).astype(np.float64)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError(f"graph_bin must be a square matrix, got {graph.shape}")
    np.fill_diagonal(graph, 0.0)
    n = graph.shape[0]
    if n < 2:
        return {
            "Density": 0.0,
            "AverageDegree": 0.0,
            "EdgeCount": 0.0,
            "ComponentCount": float(n),
            "LargestComponentSize": float(n),
        }

    if directed:
        edge_count = float(graph.sum())
        possible_edges = float(n * (n - 1))
        average_degree = float((graph.sum(axis=1) + graph.sum(axis=0)).mean())
        reach_graph = ((graph + graph.T) > 0.0).astype(np.float64)
    else:
        graph = np.maximum(graph, graph.T)
        edge_count = float(graph.sum() / 2.0)
        possible_edges = float(n * (n - 1) / 2.0)
        average_degree = float(graph.sum(axis=1).mean())
        reach_graph = graph

    visited = np.zeros(n, dtype=bool)
    component_sizes: list[int] = []
    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbor in np.where(reach_graph[node] > 0.0)[0]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(int(neighbor))
        component_sizes.append(size)

    density = edge_count / possible_edges if possible_edges > 0.0 else 0.0
    return {
        "Density": float(density),
        "AverageDegree": average_degree,
        "EdgeCount": edge_count,
        "ComponentCount": float(len(component_sizes)),
        "LargestComponentSize": float(max(component_sizes) if component_sizes else 0),
    }


def largest_component_size(graph_bin: np.ndarray) -> int:
    """Size of the largest connected component of an undirected binary graph."""
    graph = (np.asarray(graph_bin) > 0.0).astype(np.float64)
    graph = np.maximum(graph, graph.T)
    np.fill_diagonal(graph, 0.0)
    if graph.shape[0] == 0:
        return 0
    reachable = np.isfinite(_bfs_distance(graph))
    return int(reachable.sum(axis=1).max())
