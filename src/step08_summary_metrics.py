"""
Step 08: Per-subject scalar metric summary.
"""

from __future__ import annotations

import numpy as np

from pipeline_config import SPARSITY_LEVELS
from step02_connectivity import binarize, proportional_threshold
from step03_connection_strength import average_connection_strength
from step04_graph_metrics import (
    clustering_coef_bin_und,
    clustering_coef_wu,
    global_efficiency_bin,
    global_efficiency_weighted,
    graph_qc_metrics,
)


def _normalized_auc(x: list[float], y: list[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if valid.sum() < 2:
        return np.nan
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]
    span = x_arr[-1] - x_arr[0]
    if span <= 0.0:
        return np.nan
    return float(np.trapezoid(y_arr, x_arr) / span)


def compute_all_scalar_metrics(
    lofc: np.ndarray,
    hofc: np.ndarray,
    threshold: float = 0.35,
    sparsity_levels: list[float] | None = None,
) -> dict:
    """
    Compute scalar graph metrics on fixed-threshold, weighted, and fixed-density graphs.
    """
    if sparsity_levels is None:
        sparsity_levels = SPARSITY_LEVELS

    result = {}
    fc_inputs: list[tuple[str, np.ndarray, bool]] = [
        ("LOFC", lofc, False),
        ("HOFC", hofc, False),
    ]

    for fc_name, matrix, is_directed in fc_inputs:
        graph_bin = binarize(matrix, threshold)
        clustering_bin = float(clustering_coef_bin_und(graph_bin).mean())

        result[f"Eglob_bin_{fc_name}"] = global_efficiency_bin(graph_bin)
        result[f"Clustering_bin_{fc_name}"] = clustering_bin
        result[f"MeanConn_{fc_name}"] = average_connection_strength(matrix, directed=is_directed)

        clustering_weighted = float(clustering_coef_wu(matrix).mean())

        result[f"Eglob_weighted_{fc_name}"] = global_efficiency_weighted(
            matrix,
            directed=is_directed,
        )
        result[f"Clustering_weighted_{fc_name}"] = clustering_weighted

        qc = graph_qc_metrics(graph_bin, directed=is_directed)
        for key, value in qc.items():
            result[f"{key}_thr035_{fc_name}"] = value

        eglob_by_density: list[float] = []
        clustering_by_density: list[float] = []
        for density in sparsity_levels:
            sparse_graph = proportional_threshold(matrix, density=density, directed=is_directed)
            sparse_clustering = float(clustering_coef_bin_und(sparse_graph).mean())

            density_label = f"{int(round(density * 100)):02d}"
            result[f"Eglob_bin_density{density_label}_{fc_name}"] = global_efficiency_bin(
                sparse_graph
            )
            result[f"Clustering_bin_density{density_label}_{fc_name}"] = sparse_clustering
            result[f"Density_actual_density{density_label}_{fc_name}"] = graph_qc_metrics(
                sparse_graph,
                directed=is_directed,
            )["Density"]

            eglob_by_density.append(result[f"Eglob_bin_density{density_label}_{fc_name}"])
            clustering_by_density.append(sparse_clustering)

        result[f"Eglob_bin_auc_{fc_name}"] = _normalized_auc(sparsity_levels, eglob_by_density)
        result[f"Clustering_bin_auc_{fc_name}"] = _normalized_auc(
            sparsity_levels,
            clustering_by_density,
        )

    return result
