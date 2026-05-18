from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from step11_statistical_analysis import (
    _analyze_endpoint,
    _fdr_bh,
    _normality_test,
    _read_attack_curve,
    run_mixed_effects_metrics,
    run_two_way_anova_metrics,
    attack_auc,
)


def test_fdr_bh_adjusts_monotonically() -> None:
    adjusted = _fdr_bh([0.01, 0.03, 0.04])

    np.testing.assert_allclose(adjusted, [0.03, 0.04, 0.04])


def test_normality_rejects_constant_values() -> None:
    p_value, is_normal = _normality_test(pd.Series([1.0, 1.0, 1.0, 1.0]))

    assert np.isnan(p_value)
    assert is_normal is False


def test_attack_auc_uses_normalized_trapezoid_area() -> None:
    auc = attack_auc([1.0, 0.5, 0.0])

    assert auc == 0.5


def test_attack_curve_reader_handles_column_csv() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "curve.csv"
        pd.DataFrame([1.0, 0.8, 0.6]).to_csv(csv_path, index=False, header=False)

        curve = _read_attack_curve(csv_path)

    np.testing.assert_allclose(curve, [1.0, 0.8, 0.6])


def test_endpoint_analysis_includes_omnibus_before_posthoc() -> None:
    records = _analyze_endpoint(
        context={"Analysis": "Smoke", "Band": "Delta", "Metric": "MetricA"},
        group_data={
            "HC": [1.0, 1.1, 0.9, 1.05, 0.95],
            "MCI": [5.0, 5.1, 4.9, 5.05, 4.95],
            "AD": [9.0, 9.1, 8.9, 9.05, 8.95],
        },
    )

    assert records[0]["Test_Level"] == "Omnibus"
    assert any(record["Test_Level"] == "Posthoc" for record in records)


def test_two_way_anova_metrics_writes_group_band_effects() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        output_dir = base / "output"
        results_dir = base / "results"
        output_dir.mkdir()
        rows = []
        for group_idx, group in enumerate(["HC", "MCI", "AD"]):
            for band_idx, band in enumerate(["Delta", "Theta", "Alpha", "Beta"]):
                for subj in range(2):
                    rows.append(
                        {
                            "Group": group,
                            "Band": band,
                            "SubjectID": f"{group}_{band}_{subj}",
                            "MetricA": group_idx + band_idx * 0.1 + subj * 0.01,
                        }
                    )
        pd.DataFrame(rows).to_csv(output_dir / "all_scalar_metrics_summary.csv", index=False)

        result = run_two_way_anova_metrics(output_dir=output_dir, results_dir=results_dir)

        assert {"Group", "Band", "Group:Band"}.issubset(set(result["Effect"]))
        assert (results_dir / "two_way_anova_metrics.csv").exists()


def test_mixed_effects_metrics_writes_repeated_measure_effects() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        output_dir = base / "output"
        results_dir = base / "results"
        output_dir.mkdir()
        rows = []
        for group_idx, group in enumerate(["HC", "MCI", "AD"]):
            for subj in range(3):
                subject_id = f"{group}_{subj}"
                for band_idx, band in enumerate(["Delta", "Theta", "Alpha", "Beta"]):
                    rows.append(
                        {
                            "Group": group,
                            "Band": band,
                            "SubjectID": subject_id,
                            "MetricA": group_idx + band_idx * 0.1 + subj * 0.01,
                        }
                    )
        pd.DataFrame(rows).to_csv(output_dir / "all_scalar_metrics_summary.csv", index=False)

        result = run_mixed_effects_metrics(output_dir=output_dir, results_dir=results_dir)

        assert {
            "Band_within_subject",
            "Group:Band_within_subject",
            "Group_between_subject_mean",
        }.issubset(set(result["Effect"]))
        assert set(result["Model_Type"]) == {"repeated_measures_fallback"}
        assert (results_dir / "mixed_effects_metrics.csv").exists()


if __name__ == "__main__":
    test_fdr_bh_adjusts_monotonically()
    test_normality_rejects_constant_values()
    test_attack_auc_uses_normalized_trapezoid_area()
    test_attack_curve_reader_handles_column_csv()
    test_endpoint_analysis_includes_omnibus_before_posthoc()
    test_two_way_anova_metrics_writes_group_band_effects()
    test_mixed_effects_metrics_writes_repeated_measure_effects()
    print("statistical analysis smoke tests passed")
