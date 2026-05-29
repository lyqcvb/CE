"""
Step 03: Average connection strength.

This module belongs between FC construction and graph-theory metrics because
MeanConn is computed directly on the original FC matrices, before thresholding
or graph conversion.

Formula:
- LOFC/HOFC: mean of upper-triangle weights, i < j.
"""

from __future__ import annotations

import numpy as np


def average_connection_strength(matrix: np.ndarray, directed: bool = False) -> float:
    """
    Average connection weight on the original FC matrix.

    LOFC/HOFC are undirected and use the upper triangle only. Directed mode is
    retained for generic graph helpers and uses all off-diagonal elements.
    """
    weights = np.asarray(matrix, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError(f"matrix must be square, got {weights.shape}")

    n = weights.shape[0]
    if n < 2:
        return 0.0

    if directed:
        off_diagonal = ~np.eye(n, dtype=bool)
        return float(weights[off_diagonal].sum() / (n * (n - 1)))

    upper_triangle = np.triu_indices(n, 1)
    return float(weights[upper_triangle].sum() / (n * (n - 1) / 2))
