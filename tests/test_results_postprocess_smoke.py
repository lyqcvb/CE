from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from step10_results_postprocess import (
    _read_csv_vector,
    compute_attack_feature_summary,
    compute_richclub_feature_table,
    extract_targetattack_summary,
)


def test_read_csv_vector_handles_column_and_row_vectors() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        column_path = tmp_path / "column.csv"
        row_path = tmp_path / "row.csv"
        pd.DataFrame([1.0, 0.8, 0.6]).to_csv(column_path, index=False, header=False)
        pd.DataFrame([[1.0, 0.8, 0.6]]).to_csv(row_path, index=False, header=False)

        column_values = _read_csv_vector(column_path)
        row_values = _read_csv_vector(row_path)

    np.testing.assert_allclose(column_values, [1.0, 0.8, 0.6])
    np.testing.assert_allclose(row_values, [1.0, 0.8, 0.6])


def test_targetattack_summary_reads_column_curves() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        output_dir = base / "output"
        results_dir = base / "results"

        for group, offset in [("HC", 0.0), ("MCI", 0.1), ("AD", 0.2)]:
            band_dir = output_dir / "target_attack" / "lofc" / group / "1-4(Delta)"
            band_dir.mkdir(parents=True)
            for idx in range(2):
                curve = np.array([1.0, 0.8 - offset, 0.6 - offset]) - idx * 0.01
                pd.DataFrame(curve).to_csv(band_dir / f"{group}{idx}.csv", index=False, header=False)

        written = extract_targetattack_summary(
            output_dir=output_dir,
            results_dir=results_dir,
            fc_types=["lofc"],
            band_mapping={"1-4(Delta)": "Delta"},
        )

    summary = written["LOFC"]
    assert summary["Attack_Step"].tolist() == [0, 1, 2]
    np.testing.assert_allclose(summary["Delta_HC"], [0.995, 0.795, 0.595])


def test_attack_feature_summary_computes_ce_control_differences() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        output_dir = base / "output"
        results_dir = base / "results"
        curves = {
            "ce": [1.0, 0.5, 0.0],
            "degree": [1.0, 0.7, 0.2],
            "betweenness": [1.0, 0.6, 0.1],
            "random_mean": [1.0, 0.8, 0.4],
        }
        for strategy, curve in curves.items():
            path = (
                output_dir
                / "target_attack_controls"
                / strategy
                / "lofc"
                / "HC"
                / "1-4(Delta)"
            )
            path.mkdir(parents=True)
            pd.DataFrame(curve).to_csv(path / "S1.csv", index=False, header=False)

        df = compute_attack_feature_summary(output_dir=output_dir, results_dir=results_dir)

    assert df is not None
    assert "CE_vs_Random_AUC_Diff" in df.columns
    assert "AttackAUC_CE" in df.columns
    assert "EarlyDrop_1" in df.columns
    assert df.loc[0, "CE_vs_Random_AUC_Diff"] > 0.0


def test_richclub_feature_table_extracts_high_k_features() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        output_dir = base / "output"
        results_dir = base / "results"
        path = output_dir / "rich_club" / "lofc" / "HC" / "1-4(Delta)"
        path.mkdir(parents=True)
        pd.DataFrame([[0.8, 1.1, 1.3, np.nan]]).to_csv(
            path / "S1.csv",
            index=False,
            header=False,
        )

        df = compute_richclub_feature_table(
            output_dir=output_dir,
            results_dir=results_dir,
            fc_types=["lofc"],
            band_mapping={"1-4(Delta)": "Delta"},
            groups=["HC"],
        )

    assert df is not None
    assert df.loc[0, "RichClubEnhancedKCount"] == 2.0
    assert df.loc[0, "RichClubPeakK"] == 3.0


if __name__ == "__main__":
    test_read_csv_vector_handles_column_and_row_vectors()
    test_targetattack_summary_reads_column_curves()
    test_attack_feature_summary_computes_ce_control_differences()
    test_richclub_feature_table_extracts_high_k_features()
    print("results postprocess smoke tests passed")
