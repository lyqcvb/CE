from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import pipeline_config
from step09_batch_pipeline import _filter_current_summary_scope


def test_current_experiment_scope_excludes_gamma_ahofc_and_richclub() -> None:
    assert pipeline_config.BANDS == ["Delta", "Theta", "Alpha", "Beta"]
    assert "30-40(Gamma)" not in pipeline_config.BAND_LABELS
    assert pipeline_config.FC_TYPES_LOWER == ["lofc", "hofc"]
    assert pipeline_config.FC_TYPES_UPPER == ["LOFC", "HOFC"]


def test_summary_scope_filter_drops_stale_experiment_outputs() -> None:
    df = pd.DataFrame(
        [
            {
                "Group": "HC",
                "Band": "Delta",
                "SubjectID": "S1.set",
                "Eglob_bin_LOFC": 0.1,
                "Eglob_bin_aHOFC": 0.2,
                "RichClub_auc_LOFC": 0.3,
            },
            {
                "Group": "HC",
                "Band": "Gamma",
                "SubjectID": "S1.set",
                "Eglob_bin_LOFC": 0.4,
                "Eglob_bin_aHOFC": 0.5,
                "RichClub_auc_LOFC": 0.6,
            },
        ]
    )

    filtered = _filter_current_summary_scope(
        df,
        allowed_columns=["Group", "Band", "SubjectID", "Eglob_bin_LOFC"],
    )

    assert filtered["Band"].tolist() == ["Delta"]
    assert "Eglob_bin_LOFC" in filtered.columns
    assert "Eglob_bin_aHOFC" not in filtered.columns
    assert "RichClub_auc_LOFC" not in filtered.columns
