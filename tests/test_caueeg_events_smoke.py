from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from step14_caueeg_events import assign_groups, build_spans, correct_pause_offsets  # noqa: E402

FS = 200


def test_pause_correction_shifts_events_after_resume():
    events = [
        [0, "Start Recording"],
        [1000, "Paused"],
        [1600, "Recording Resumed"],
        [2000, "Eyes Closed"],
        [5000, "Paused"],
        [5400, "Recording Resumed"],
        [6000, "Eyes Open"],
    ]
    corrected, boundaries = correct_pause_offsets(events)
    times = dict((label + str(i), t) for i, (t, label) in enumerate(corrected))
    assert times["Eyes Closed3"] == 1400
    assert times["Eyes Open6"] == 6000 - 600 - 400
    assert boundaries == [1000, 4400]


def test_spans_split_at_pause_and_exclude_photic_hv_and_artifacts():
    n = 400 * FS
    events = [
        [0, "Start Recording"],
        [10 * FS, "Eyes Closed"],
        [40 * FS, "Paused"],
        [50 * FS, "Recording Resumed"],      # corrected boundary at 40 s
        [70 * FS, "Move"],                   # corrected 60 s, +/-1 s removed
        [100 * FS, "Eyes Open"],             # corrected 90 s
        [110 * FS, "Passive eye close"],     # corrected 100 s
        [120 * FS, "Photic On - 3.0 Hz"],
        [130 * FS, "Photic Off"],
        [150 * FS, "HV - Dur: 20.0 sec. - On"],
        [170 * FS, "HV - Off"],              # corrected 160 s, +30 s buffer -> 190 s
        [250 * FS, "Eyes Open"],             # corrected 240 s
    ]
    spans = build_spans(events, n)
    secs = [(a / FS, b / FS) for a, b in spans["ec_spans"]]
    assert secs == [(10, 40), (40, 59), (61, 90), (100, 110), (120, 140), (190, 240)]
    assert spans["has_hv"] and spans["has_photic"]
    assert spans["pause_boundaries"] == [40 * FS]


def test_sleep_excluded_until_next_eye_event_and_record_tail():
    n = 100 * FS
    events = [[0, "Eyes Closed"], [20 * FS, "Sleep"], [50 * FS, "Eyes Closed"]]
    spans = build_spans(events, n)
    secs = [(a / FS, b / FS) for a, b in spans["ec_spans"]]
    assert secs == [(0, 20), (50, 95)]


def test_group_assignment_rules():
    df = pd.DataFrame({
        "dementia_cls": ["Normal", "Normal", "MCI", "MCI", "Dementia", "Dementia", "Dementia"],
        "symptom": [["normal"], ["normal", "smi"], ["mci", "mci_amnestic"], ["mci", "mci_amnestic", "mci_vascular"],
                    ["dementia", "ad", "load"], ["dementia", "vd"], ["dementia", "ad", "ad_vd_mixed"]],
    })
    assert list(assign_groups(df).fillna("NA")) == ["HC", "NA", "MCI", "NA", "AD", "NA", "NA"]
