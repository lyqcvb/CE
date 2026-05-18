"""
Step 02: Functional connectivity matrices.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert


def _validate_channel_time_data(eeg_data: np.ndarray) -> np.ndarray:
    data = np.asarray(eeg_data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"EEG data must be 2D as channels x samples, got {data.shape}")
    if data.shape[0] < 2 or data.shape[1] < 2:
        raise ValueError(f"EEG data is too small for correlation: {data.shape}")
    return data


def _validate_connectivity_matrix(matrix: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got {arr.shape}")
    if arr.shape[0] < 3:
        raise ValueError(f"{name} needs at least 3 nodes for topographical FC")
    return arr


def _absolute_pearson_matrix(row_signals: np.ndarray) -> np.ndarray:
    """Return absolute Pearson correlation between rows."""
    corr = np.corrcoef(row_signals)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.abs(corr)
    np.fill_diagonal(corr, 1.0)
    return corr


def _signed_pearson_matrix(row_signals: np.ndarray) -> np.ndarray:
    """Return signed Pearson correlation between rows."""
    corr = np.corrcoef(row_signals)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def compute_envelope_correlation(eeg_data: np.ndarray, absolute: bool = True) -> np.ndarray:
    """
    Compute low-order FC as Hilbert envelope correlation.
    """
    data = _validate_channel_time_data(eeg_data)
    envelopes = np.abs(hilbert(data, axis=1))
    if not absolute:
        return _signed_pearson_matrix(envelopes)
    return _absolute_pearson_matrix(envelopes)


def compute_lofc(eeg_data: np.ndarray) -> np.ndarray:
    """Compute low-order FC as Hilbert envelope correlation."""
    return compute_envelope_correlation(eeg_data)


def compute_signed_lofc(eeg_data: np.ndarray) -> np.ndarray:
    """Compute signed Hilbert envelope correlation for LOFC sensitivity checks."""
    return compute_envelope_correlation(eeg_data, absolute=False)


def fisher_z_transform_connectivity(matrix: np.ndarray) -> np.ndarray:
    """
    Apply Fisher z transform to FC values before topographical correlation.
    """
    eps = np.finfo(np.float64).eps
    clipped = np.clip(matrix, -1.0 + eps, 1.0 - eps)
    return np.arctanh(clipped)


def _topographical_profile(
    z_matrix: np.ndarray,
    node_index: int,
    excluded_nodes: set[int],
) -> np.ndarray:
    mask = np.ones(z_matrix.shape[0], dtype=bool)
    mask[list(excluded_nodes)] = False
    return z_matrix[mask, node_index]


def _absolute_profile_correlation(profile_a: np.ndarray, profile_b: np.ndarray) -> float:
    if profile_a.size < 2 or profile_b.size < 2:
        return 0.0
    if np.allclose(profile_a, profile_a[0]) or np.allclose(profile_b, profile_b[0]):
        return 0.0

    r = np.corrcoef(profile_a, profile_b)[0, 1]
    if not np.isfinite(r):
        return 0.0
    return float(abs(r))


def compute_topographical_profile_correlation(
    source_fc: np.ndarray,
    target_fc: np.ndarray,
    *,
    symmetric_output: bool,
) -> np.ndarray:
    """
    Correlate Fisher-z topographical profiles between two FC matrices.
    """
    source = _validate_connectivity_matrix(source_fc, "source_fc")
    target = _validate_connectivity_matrix(target_fc, "target_fc")
    if source.shape != target.shape:
        raise ValueError(
            f"source_fc and target_fc must have the same shape: {source.shape} != {target.shape}"
        )

    n_nodes = source.shape[0]
    source_z = fisher_z_transform_connectivity(source)
    target_z = fisher_z_transform_connectivity(target)
    out = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    if symmetric_output:
        for i in range(n_nodes - 1):
            for j in range(i + 1, n_nodes):
                excluded = {i, j}
                source_profile = _topographical_profile(source_z, i, excluded)
                target_profile = _topographical_profile(target_z, j, excluded)
                value = _absolute_profile_correlation(source_profile, target_profile)
                out[i, j] = value
                out[j, i] = value
    else:
        for i in range(n_nodes):
            for j in range(n_nodes):
                excluded = {i, j}
                source_profile = _topographical_profile(source_z, i, excluded)
                target_profile = _topographical_profile(target_z, j, excluded)
                out[i, j] = _absolute_profile_correlation(source_profile, target_profile)

    return out


def compute_topographical_hofc(lofc: np.ndarray) -> np.ndarray:
    """
    Compute high-order FC from LOFC topographical profiles.
    """
    return compute_topographical_profile_correlation(
        lofc,
        lofc,
        symmetric_output=True,
    )


def compute_hofc(lofc: np.ndarray) -> np.ndarray:
    """Compute high-order FC from LOFC topographical profiles."""
    return compute_topographical_hofc(lofc)


def compute_associated_hofc(lofc: np.ndarray, hofc: np.ndarray) -> np.ndarray:
    """
    Compute associated HOFC by correlating LOFC profiles with HOFC profiles.
    """
    return compute_topographical_profile_correlation(
        lofc,
        hofc,
        symmetric_output=False,
    )


def compute_ahofc(lofc: np.ndarray, hofc: np.ndarray) -> np.ndarray:
    """Compute associated HOFC from LOFC and HOFC topographical profiles."""
    return compute_associated_hofc(lofc, hofc)


def binarize(matrix: np.ndarray, threshold: float = 0.35) -> np.ndarray:
    """
    Binarize a connectivity matrix and remove self-connections.
    """
    graph = (matrix >= threshold).astype(np.float64)
    np.fill_diagonal(graph, 0)
    return graph


def proportional_threshold(
    matrix: np.ndarray,
    density: float,
    directed: bool = False,
) -> np.ndarray:
    """
    Binarize by fixed density, retaining the strongest off-diagonal edges.

    For undirected matrices, only the upper triangle is ranked and the returned
    graph is symmetrized. For directed matrices, all off-diagonal ordered edges
    are ranked independently.
    """
    if not 0 < density <= 1:
        raise ValueError("density must be in (0, 1].")

    arr = np.asarray(matrix, dtype=np.float64).copy()
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"matrix must be square, got {arr.shape}")

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(arr, -np.inf)
    n = arr.shape[0]
    graph = np.zeros_like(arr, dtype=np.float64)

    if directed:
        rows, cols = np.where(~np.eye(n, dtype=bool))
    else:
        rows, cols = np.triu_indices(n, k=1)

    weights = arr[rows, cols]
    valid = np.isfinite(weights)
    rows = rows[valid]
    cols = cols[valid]
    weights = weights[valid]
    if weights.size == 0:
        return graph

    edge_count = max(1, int(np.ceil(weights.size * density)))
    order = np.lexsort((cols, rows, -weights))
    chosen = order[:edge_count]
    graph[rows[chosen], cols[chosen]] = 1.0
    if not directed:
        graph[cols[chosen], rows[chosen]] = 1.0

    np.fill_diagonal(graph, 0.0)
    return graph
