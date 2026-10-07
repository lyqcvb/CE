"""
Step 14: CAUEEG event correction and candidate pool.

CAUEEG event JSON stores ``[sample_index, label]`` pairs at 200 Hz. The EDF
files omit paused time, while event indices keep counting through pauses, so
every event after ``Recording Resumed`` must be shifted back by the cumulative
pause length. Without this correction eyes-closed spans in ~25% of records are
misplaced (median 37 s).

Outputs
-------
data/caueeg/events_corrected/{serial}.json   corrected spans per record
results/caueeg/candidate_pool.csv            serial, age, group, usable EC time, flags
results/caueeg/preprocess_serials.txt        serial list only (group-blind input for step15)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import caueeg_config as cfg

_ARTIFACT_LABELS = (
    "move", "artifact", "swallowing", "cough", "talk", "chewing", "couch",
    "jerking", "mouth movement", "sweating", "eye blinking", "snoring",
)
_CHECK_PATTERN = re.compile(r"(check|recheck|new montage|new low filters|new sensitivity|power failure)")
_CHECK_PAD_SEC = 5.0


def read_edf_n_samples(edf_path: Path) -> int:
    """Number of samples per EEG channel from the EDF header (1-s data records at 200 Hz)."""
    with open(edf_path, "rb") as fh:
        header = fh.read(256)
        n_signals = int(header[252:256])
        n_records = int(header[236:244])
        duration = float(header[244:252])
        signal_header = fh.read(256 * n_signals)
    offset = (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80) * n_signals
    per_record = [
        int(signal_header[offset + i * 8: offset + (i + 1) * 8].decode("ascii").strip())
        for i in range(n_signals)
    ]
    if duration != 1.0:
        raise ValueError(f"{edf_path}: expected 1-s data records, got {duration}")
    return int(n_records * per_record[0])


def correct_pause_offsets(events: list[list]) -> tuple[list[tuple[int, str]], list[int]]:
    """
    Shift events after each ``Recording Resumed`` back by the cumulative pause.

    Returns corrected ``(sample, label)`` events (in original order) and the
    corrected sample positions of resume points, which are signal
    discontinuities in the EDF.
    """
    corrected: list[tuple[int, str]] = []
    boundaries: list[int] = []
    offset = 0
    paused_at: int | None = None
    for raw_t, label in events:
        t = int(raw_t)
        low = str(label).lower()
        if low.startswith("paused"):
            paused_at = t
            corrected.append((t - offset, label))
            continue
        if "recording resumed" in low:
            if paused_at is not None and t >= paused_at:
                boundaries.append(paused_at - offset)
                offset += t - paused_at
            paused_at = None
            corrected.append((t - offset, label))
            continue
        corrected.append((t - offset, label))
    return corrected, boundaries


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _subtract(spans: list[tuple[int, int]], excluded: list[tuple[int, int]]) -> list[tuple[int, int]]:
    result = []
    excluded = _merge(excluded)
    for a, b in spans:
        cur = a
        for ea, eb in excluded:
            if eb <= cur or ea >= b:
                continue
            if ea > cur:
                result.append((cur, ea))
            cur = max(cur, eb)
            if cur >= b:
                break
        if cur < b:
            result.append((cur, b))
    return [(a, b) for a, b in result if b > a]


def _split_at(spans: list[tuple[int, int]], cuts: list[int]) -> list[tuple[int, int]]:
    out = []
    for a, b in spans:
        points = [a] + sorted(c for c in cuts if a < c < b) + [b]
        out.extend((points[i], points[i + 1]) for i in range(len(points) - 1))
    return out


def _is_eyes_closed(low: str) -> bool:
    return "eyes closed" in low or "passive eye close" in low


def _is_eyes_open(low: str) -> bool:
    return "eyes open" in low or "passive eye open" in low


def _is_hv_on(low: str) -> bool:
    return low.startswith("hv") and low.rstrip().endswith("on") and "post" not in low


def build_spans(events: list[list], n_samples: int, sfreq: float = cfg.SFREQ) -> dict:
    """
    Derive eyes-closed analysis spans and ICA-usable spans from raw events.

    Eyes-closed spans start at Eyes Closed / passive eye close and end at the
    next eyes-open event. Photic stimulation, hyperventilation (+30 s), sleep
    or drowsiness (until the next eye-state event), artifact annotations
    (+/-1 s), electrode checks (+/-5 s) and the last 5 s are excluded. Spans
    are split at pause boundaries because the EDF signal is discontinuous there.
    """
    corrected, boundaries = correct_pause_offsets(events)
    end_limit = n_samples - int(cfg.RECORD_TAIL_SEC * sfreq)
    pad = int(cfg.EVENT_ARTIFACT_PAD_SEC * sfreq)
    check_pad = int(_CHECK_PAD_SEC * sfreq)
    hv_buffer = int(cfg.HV_BUFFER_SEC * sfreq)

    ec_spans: list[tuple[int, int]] = []
    excluded: list[tuple[int, int]] = []
    ec_start: int | None = None
    photic_start: int | None = None
    hv_start: int | None = None
    hv_duration: int | None = None
    sleep_start: int | None = None
    flags = {"has_hv": False, "has_sleep": False, "has_photic": False}

    for t, label in corrected:
        low = str(label).lower().strip()
        if _is_eyes_closed(low) or _is_eyes_open(low):
            if sleep_start is not None:
                excluded.append((sleep_start, t))
                sleep_start = None
        if _is_eyes_closed(low):
            if ec_start is None:
                ec_start = t
        elif _is_eyes_open(low):
            if ec_start is not None:
                ec_spans.append((ec_start, t))
                ec_start = None
        elif low.startswith("photic on"):
            flags["has_photic"] = True
            if photic_start is None:
                photic_start = t
        elif low.startswith("photic off"):
            if photic_start is not None:
                excluded.append((photic_start, t))
                photic_start = None
        elif _is_hv_on(low):
            flags["has_hv"] = True
            hv_start = t
            match = re.search(r"dur:\s*([\d.]+)", low)
            hv_duration = int(float(match.group(1)) * sfreq) if match else None
        elif low.startswith("hv - off") or low.startswith("hv off"):
            if hv_start is not None:
                excluded.append((hv_start, t + hv_buffer))
                hv_start = None
        elif "sleep" in low or "drowsy" in low or "snoring" in low:
            flags["has_sleep"] = True
            if sleep_start is None:
                sleep_start = t
        if any(key in low for key in _ARTIFACT_LABELS):
            excluded.append((t - pad, t + pad))
        elif _CHECK_PATTERN.search(low) and t > 0:
            excluded.append((t - check_pad, t + check_pad))

    if ec_start is not None:
        ec_spans.append((ec_start, n_samples))
    if photic_start is not None:
        excluded.append((photic_start, n_samples))
    if hv_start is not None:
        hv_end = hv_start + hv_duration if hv_duration is not None else n_samples
        excluded.append((hv_start, min(n_samples, hv_end + hv_buffer)))
    if sleep_start is not None:
        excluded.append((sleep_start, n_samples))
    excluded.append((end_limit, n_samples))

    ec_spans = [(max(0, a), min(b, n_samples)) for a, b in ec_spans]
    ec_clean = _split_at(_subtract(ec_spans, excluded), boundaries)
    # ICA may use eyes-open data too (helps isolate blinks), but not photic/HV/sleep.
    usable = _split_at(_subtract([(0, n_samples)], excluded), boundaries)
    last_event = corrected[-1][0] if corrected else 0
    return {
        "n_samples": int(n_samples),
        "pause_boundaries": [int(b) for b in boundaries],
        "ec_spans": [[int(a), int(b)] for a, b in ec_clean],
        "usable_spans": [[int(a), int(b)] for a, b in usable],
        "excluded": [[int(a), int(b)] for a, b in _merge(excluded)],
        "last_event_corrected": int(last_event),
        "event_overrun": int(last_event - n_samples),
        **flags,
    }


def _span_seconds(spans: list[list[int]], sfreq: float = cfg.SFREQ) -> float:
    return float(sum(b - a for a, b in spans) / sfreq)


def assign_groups(annotation: pd.DataFrame) -> pd.Series:
    """HC = Normal without SMI; MCI = amnestic MCI without vascular; AD = AD without VD/mixed."""
    sym = annotation["symptom"].apply(lambda s: set(s or []))
    cls = annotation["dementia_cls"]
    group = pd.Series(pd.NA, index=annotation.index, dtype="object")
    group[(cls == "Normal") & ~sym.apply(lambda s: "smi" in s)] = "HC"
    group[(cls == "MCI") & sym.apply(lambda s: "mci_amnestic" in s and "mci_vascular" not in s)] = "MCI"
    group[(cls == "Dementia") & sym.apply(lambda s: "ad" in s and not ({"vd", "ad_vd_mixed"} & s))] = "AD"
    return group


def load_annotation(root: Path) -> pd.DataFrame:
    annotation = pd.DataFrame(json.loads((root / "annotation.json").read_text(encoding="utf-8"))["data"])
    task = json.loads((root / "dementia-no-overlap.json").read_text(encoding="utf-8"))
    rows = []
    for split in ("train_split", "validation_split", "test_split"):
        for item in task[split]:
            rows.append({"serial": item["serial"], "dementia_cls": item["class_name"], "split": split})
    labels = pd.DataFrame(rows)
    df = annotation.merge(labels, on="serial", how="left")
    df["group"] = assign_groups(df)
    return df


def run(args: argparse.Namespace) -> pd.DataFrame:
    paths = cfg.paths_from_args(args)
    root = Path(args.caueeg_root)
    paths.events.mkdir(parents=True, exist_ok=True)
    df = load_annotation(root)
    lo, hi = cfg.AGE_RANGE
    df = df[df["group"].notna() & df["age"].between(lo, hi)].copy()

    records = []
    for serial in df["serial"]:
        events = json.loads((root / "event" / f"{serial}.json").read_text(encoding="utf-8"))
        n_samples = read_edf_n_samples(root / "signal" / "edf" / f"{serial}.edf")
        spans = build_spans(events, n_samples)
        (paths.events / f"{serial}.json").write_text(json.dumps(spans), encoding="utf-8")
        records.append({
            "serial": serial,
            "duration_s": n_samples / cfg.SFREQ,
            "ec_usable_s": _span_seconds(spans["ec_spans"]),
            "ica_usable_s": _span_seconds(spans["usable_spans"]),
            "n_pauses": len(spans["pause_boundaries"]),
            "event_overrun_s": spans["event_overrun"] / cfg.SFREQ,
            "has_hv": spans["has_hv"],
            "has_sleep": spans["has_sleep"],
            "has_photic": spans["has_photic"],
        })
    pool = df[["serial", "age", "group", "dementia_cls", "split"]].merge(pd.DataFrame(records), on="serial")
    pool["symptom"] = df.set_index("serial").loc[pool["serial"], "symptom"].apply(";".join).values
    pool["events_aligned"] = pool["event_overrun_s"] <= 6.0
    pool["preprocess"] = pool["events_aligned"] & (pool["ec_usable_s"] >= cfg.N_EPOCHS * cfg.EPOCH_SEC)
    pool.to_csv(paths.results / "candidate_pool.csv", index=False, encoding="utf-8-sig")
    serials = pool.loc[pool["preprocess"], "serial"].sort_values()
    (paths.results / "preprocess_serials.txt").write_text("\n".join(serials) + "\n", encoding="utf-8")

    summary = pool.groupby("group").agg(
        n=("serial", "size"),
        to_preprocess=("preprocess", "sum"),
        misaligned=("events_aligned", lambda s: int((~s).sum())),
        with_pause=("n_pauses", lambda s: int((s > 0).sum())),
        ec_usable_median=("ec_usable_s", "median"),
        age_mean=("age", "mean"),
    )
    print(summary.to_string())
    return pool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CAUEEG 事件暂停校正与候选池")
    return cfg.add_common_args(parser)


if __name__ == "__main__":
    run(build_parser().parse_args())
