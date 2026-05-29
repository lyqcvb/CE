from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from step08_summary_metrics import compute_all_scalar_metrics


def test_scalar_metrics_include_weighted_qc_and_density_auc() -> None:
    rng = np.random.default_rng(42)
    lofc = rng.random((5, 5))
    lofc = (lofc + lofc.T) / 2.0
    np.fill_diagonal(lofc, 1.0)
    hofc = rng.random((5, 5))
    hofc = (hofc + hofc.T) / 2.0
    np.fill_diagonal(hofc, 1.0)
    metrics = compute_all_scalar_metrics(
        lofc,
        hofc,
        sparsity_levels=[0.2, 0.4],
    )

    for key in [
        "Eglob_weighted_LOFC",
        "Eglob_bin_auc_HOFC",
        "Clustering_bin_auc_LOFC",
        "Density_thr035_HOFC",
    ]:
        assert key in metrics
        assert np.isfinite(metrics[key])
