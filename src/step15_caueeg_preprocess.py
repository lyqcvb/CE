"""
Step 15: Pre-specified, group-blind CAUEEG preprocessing with MNE.

Input is only the serial list written by step14 (no diagnosis labels), the raw
EDF, and the corrected spans in data/caueeg/events_corrected.

Per record:
1. Read the 19 EEG channels (EDF unit must be uV), standard_1020 montage.
2. Bad channels on usable data: flat (SD < 0.5 uV), robust log-variance
   |z| > 5, or median correlation with the 4 nearest channels < 0.4.
3. Average reference excluding bad channels, 60 Hz notch.
4. ICA on a 1 Hz high-passed copy of all usable data (eyes open + closed, no
   photic/HV/sleep): Picard (extended, non-orthogonal), n_components = rank.
   ICLabel; drop Muscle/Eye/Heart/Line Noise/Channel Noise components with
   probability >= 0.80.
5. Apply ICA to 1-45 Hz data, spherical-spline interpolate bad channels,
   re-reference to average.
6. Non-overlapping 2 s epochs inside eyes-closed spans (>= 1 s from span
   edges), reject peak-to-peak > 150 uV, keep the first 50 clean epochs.

Outputs
-------
data/caueeg/preprocessed/{serial}.npz   span data (uV, float32) + epoch positions
results/caueeg/preproc_logs/{serial}.json
results/caueeg/preproc_log.csv
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis

import caueeg_config as cfg

UV = 1e-6


def load_raw(edf_path: Path):
    import mne

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    picks = [f"{name}-AVG" for name in cfg.EDF_CHANNELS]
    missing = [name for name in picks if name not in raw.ch_names]
    if missing:
        raise ValueError(f"missing channels {missing}")
    units = getattr(raw, "_orig_units", {}) or {}
    bad_units = {name: units.get(name) for name in picks if units.get(name) not in ("uV", "µV", "μV")}
    if bad_units:
        raise ValueError(f"unexpected EDF units {bad_units}")
    raw.pick(picks)
    raw.rename_channels(dict(zip(picks, cfg.CHANNELS)))
    raw.set_channel_types({name: "eeg" for name in cfg.CHANNELS})
    raw.set_montage("standard_1020", verbose="ERROR")
    return raw


def _concat(data: np.ndarray, spans: list[list[int]]) -> np.ndarray:
    pieces = [data[:, a:b] for a, b in spans if b > a]
    return np.concatenate(pieces, axis=1) if pieces else np.zeros((data.shape[0], 0))


def _window_view(data: np.ndarray, width: int) -> np.ndarray:
    n_windows = data.shape[1] // width
    return data[:, : n_windows * width].reshape(data.shape[0], n_windows, width)


def detect_bad_channels(data_uv: np.ndarray, positions: np.ndarray, sfreq: float = cfg.SFREQ) -> dict[str, list[int]]:
    """
    Flag bad channels on 1-45 Hz eyes-closed data (uV) with blink-robust rules.

    Amplitude uses the median of 2 s window SDs (robust to occasional
    artifacts). Correlation is re-referenced without amplitude outliers and
    uses the median across windows of the maximum |r| with the 4 nearest
    channels, similar to the PREP correlation criterion.
    """
    windows = _window_view(data_uv, int(cfg.EPOCH_SEC * sfreq))
    robust_sd = np.median(windows.std(axis=2), axis=1)
    flat = np.where(robust_sd < cfg.BAD_FLAT_SD_UV)[0].tolist()

    logsd = np.log(np.maximum(robust_sd, 1e-12))
    mad = np.median(np.abs(logsd - np.median(logsd))) * 1.4826
    z = (logsd - np.median(logsd)) / (mad if mad > 0 else 1.0)
    high_var = np.where(np.abs(z) > cfg.BAD_LOGVAR_Z)[0].tolist()

    outliers = set(flat) | set(high_var)
    good = np.array([i for i in range(data_uv.shape[0]) if i not in outliers])
    reref = windows - windows[good].mean(axis=0, keepdims=True)
    dist = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    dist[:, list(outliers)] = np.inf
    low_corr = []
    for idx in range(data_uv.shape[0]):
        if idx in outliers:
            continue
        neighbors = np.argsort(dist[idx])[: cfg.BAD_NEIGHBORS]
        x = reref[idx] - reref[idx].mean(axis=1, keepdims=True)
        y = reref[neighbors] - reref[neighbors].mean(axis=2, keepdims=True)
        num = (x[None] * y).sum(axis=2)
        den = np.sqrt((x**2).sum(axis=1))[None] * np.sqrt((y**2).sum(axis=2))
        r = np.abs(num / np.maximum(den, 1e-12))
        if np.median(r.max(axis=0)) < cfg.BAD_NEIGHBOR_CORR:
            low_corr.append(idx)
    return {"flat": flat, "high_variance": high_var, "low_neighbor_corr": low_corr}


def _exclusion_annotations(raw, usable_spans: list[list[int]]):
    import mne

    sfreq = raw.info["sfreq"]
    n = raw.n_times
    onsets, durations = [], []
    cursor = 0
    for a, b in sorted(usable_spans):
        if a > cursor:
            onsets.append(cursor / sfreq)
            durations.append((a - cursor) / sfreq)
        cursor = max(cursor, b)
    if cursor < n:
        onsets.append(cursor / sfreq)
        durations.append((n - cursor) / sfreq)
    return mne.Annotations(onsets, durations, ["BAD_excluded"] * len(onsets))


def select_epochs(data_uv: np.ndarray, ec_spans: list[list[int]], sfreq: float) -> tuple[list[tuple[int, int]], int]:
    """First N clean 2 s epochs in time order. Returns [(start, span_index)] and candidate count."""
    width = int(cfg.EPOCH_SEC * sfreq)
    edge = int(cfg.EPOCH_EDGE_SEC * sfreq)
    chosen: list[tuple[int, int]] = []
    candidates = 0
    for span_idx, (a, b) in enumerate(ec_spans):
        start = a + edge
        while start + width <= b - edge:
            candidates += 1
            seg = data_uv[:, start:start + width]
            if np.ptp(seg, axis=1).max() <= cfg.EPOCH_PTP_UV and len(chosen) < cfg.N_EPOCHS:
                chosen.append((start, span_idx))
            start += width
    return chosen, candidates


def _band_ratio(freqs: np.ndarray, psd: np.ndarray, num: tuple[float, float], den: tuple[float, float]) -> float:
    num_mask = (freqs >= num[0]) & (freqs < num[1])
    den_mask = (freqs >= den[0]) & (freqs < den[1])
    return float(psd[..., num_mask].sum() / max(psd[..., den_mask].sum(), 1e-30))


def preprocess_record(serial: str, caueeg_root: Path, project_root: Path) -> dict:
    import mne
    from mne.preprocessing import ICA
    from mne_icalabel.iclabel import iclabel_label_components

    mne.set_log_level("ERROR")
    paths = cfg.CaueegPaths(project_root)
    spans = json.loads((paths.events / f"{serial}.json").read_text(encoding="utf-8"))
    raw = load_raw(caueeg_root / "signal" / "edf" / f"{serial}.edf")
    sfreq = raw.info["sfreq"]
    if abs(sfreq - cfg.SFREQ) > 1e-6 or raw.n_times != spans["n_samples"]:
        raise ValueError(f"unexpected sfreq/length {sfreq} {raw.n_times} vs {spans['n_samples']}")
    log: dict = {"serial": serial, "status": "ok"}

    # Line noise before any filtering (recording-condition QC).
    raw_ec = _concat(raw.get_data() / UV, spans["ec_spans"])
    freqs, psd = welch(raw_ec, fs=sfreq, nperseg=int(4 * sfreq))
    # Mean PSD density at 59-61 Hz relative to 55-65 Hz (1 = no line peak).
    log["line60_ratio_raw"] = _band_ratio(freqs, psd, (59.0, 61.0), (55.0, 65.0)) * (10.0 / 2.0)

    # Bad channels on 1-45 Hz eyes-closed data (blinks would inflate frontal SD).
    probe = raw.copy().filter(*cfg.ANALYSIS_BAND, verbose="ERROR")
    probe_ec = _concat(probe.get_data() / UV, spans["ec_spans"])
    positions = np.array([raw.info["chs"][i]["loc"][:3] for i in range(cfg.N_CHANNELS)])
    rules = detect_bad_channels(probe_ec, positions, sfreq)
    bad_idx = sorted(set(sum(rules.values(), [])))
    bads = [cfg.CHANNELS[i] for i in bad_idx]
    log["bad_channels"] = bads
    log["bad_rules"] = {key: [cfg.CHANNELS[i] for i in val] for key, val in rules.items()}
    log["n_bad_channels"] = len(bads)
    if len(bads) > cfg.MAX_BAD_CHANNELS:
        log["status"] = "too_many_bad_channels"
        return log

    raw.info["bads"] = bads
    raw.set_eeg_reference("average", verbose="ERROR")
    raw.notch_filter(cfg.LINE_FREQ, verbose="ERROR")

    # ICA on 1 Hz high-passed usable data.
    raw_ica = raw.copy().filter(cfg.ICA_FIT_HIGHPASS, None, verbose="ERROR")
    raw_ica.set_annotations(_exclusion_annotations(raw_ica, spans["usable_spans"]))
    n_good = cfg.N_CHANNELS - len(bads)
    ica = ICA(
        n_components=n_good - 1,
        method="picard",
        fit_params={"ortho": False, "extended": True},
        max_iter="auto",
        random_state=cfg.SEED,
    )
    ica.fit(raw_ica, reject={"eeg": cfg.ICA_REJECT_UV * UV}, reject_by_annotation=True, verbose="ERROR")
    proba = iclabel_label_components(raw_ica, ica, inplace=False)
    reject_cols = [cfg.ICLABEL_CLASSES.index(name) for name in cfg.ICLABEL_REJECT]
    exclude = [int(i) for i in range(proba.shape[0]) if proba[i, reject_cols].max() >= cfg.ICLABEL_THRESHOLD]
    top = proba.argmax(axis=1)
    log["n_ica_components"] = int(proba.shape[0])
    log["ica_excluded"] = exclude
    log["n_ica_excluded"] = len(exclude)
    log["ica_excluded_labels"] = [cfg.ICLABEL_CLASSES[int(top[i])] for i in exclude]
    log["iclabel_proba"] = np.round(proba, 4).tolist()
    log["n_brain_components"] = int((top == 0).sum())
    for name in ("Eye", "Muscle", "Heart"):
        col = cfg.ICLABEL_CLASSES.index(name)
        log[f"n_excluded_{name.lower()}"] = int(sum(proba[i, col] >= cfg.ICLABEL_THRESHOLD for i in exclude))

    # Analysis data.
    raw.filter(*cfg.ANALYSIS_BAND, verbose="ERROR")
    ica.apply(raw, exclude=exclude, verbose="ERROR")
    if bads:
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")
    raw.set_eeg_reference("average", verbose="ERROR")
    data_uv = raw.get_data() / UV

    chosen, candidates = select_epochs(data_uv, spans["ec_spans"], sfreq)
    log["n_candidate_epochs"] = candidates
    log["n_clean_epochs_selected"] = len(chosen)
    log["epoch_reject_fraction"] = float(1.0 - _count_clean(data_uv, spans["ec_spans"], sfreq) / max(candidates, 1))
    if len(chosen) < cfg.N_EPOCHS:
        log["status"] = "insufficient_epochs"
        return log

    width = int(cfg.EPOCH_SEC * sfreq)
    margin = int(cfg.SPAN_MARGIN_SEC * sfreq)
    span_ids = sorted({span for _, span in chosen})
    bounds, pieces = [], []
    for span in span_ids:
        a, b = spans["ec_spans"][span]
        starts = [s for s, sp in chosen if sp == span]
        lo = max(a, min(starts) - margin)
        hi = min(b, max(starts) + width + margin)
        bounds.append((lo, hi))
        pieces.append(data_uv[:, lo:hi].astype(np.float32))
    epoch_starts = np.array([s for s, _ in chosen], dtype=np.int64)
    epoch_piece = np.array([span_ids.index(sp) for _, sp in chosen], dtype=np.int64)
    offsets = np.r_[0, np.cumsum([p.shape[1] for p in pieces])]
    np.savez_compressed(
        paths.preprocessed / f"{serial}.npz",
        data=np.concatenate(pieces, axis=1),
        piece_offsets=offsets,
        piece_bounds=np.array(bounds, dtype=np.int64),
        epoch_starts=epoch_starts,
        epoch_piece=epoch_piece,
        sfreq=sfreq,
        channels=np.array(cfg.CHANNELS),
    )

    # Residual-artifact QC on the selected epochs.
    epochs = np.stack([data_uv[:, s:s + width] for s in epoch_starts])
    freqs, psd = welch(epochs, fs=sfreq, nperseg=width, axis=-1)
    log["emg_ratio"] = _band_ratio(freqs, psd.mean(axis=0), (30.0, 45.0), (1.0, 45.0))
    log["kurtosis_mean"] = float(np.mean(kurtosis(epochs, axis=-1)))
    log["ptp_median_uv"] = float(np.median(np.ptp(epochs, axis=-1)))
    log["n_ec_pieces"] = len(span_ids)
    log["selected_span_s"] = float((epoch_starts[-1] - epoch_starts[0]) / sfreq + cfg.EPOCH_SEC)
    return log


def _count_clean(data_uv: np.ndarray, ec_spans: list[list[int]], sfreq: float) -> int:
    width = int(cfg.EPOCH_SEC * sfreq)
    edge = int(cfg.EPOCH_EDGE_SEC * sfreq)
    count = 0
    for a, b in ec_spans:
        start = a + edge
        while start + width <= b - edge:
            if np.ptp(data_uv[:, start:start + width], axis=1).max() <= cfg.EPOCH_PTP_UV:
                count += 1
            start += width
    return count


def _run_one(serial: str, caueeg_root: Path, project_root: Path) -> dict:
    paths = cfg.CaueegPaths(project_root)
    try:
        log = preprocess_record(serial, caueeg_root, project_root)
    except Exception as exc:  # keep the batch going, record the failure
        log = {"serial": serial, "status": f"error: {exc}", "traceback": traceback.format_exc()}
    (paths.preproc_logs / f"{serial}.json").write_text(json.dumps(log, ensure_ascii=False), encoding="utf-8")
    return log


def summarize_logs(paths: cfg.CaueegPaths) -> pd.DataFrame:
    rows = []
    for path in sorted(paths.preproc_logs.glob("*.json")):
        log = json.loads(path.read_text(encoding="utf-8"))
        row = {k: v for k, v in log.items() if k not in {"iclabel_proba", "traceback", "bad_rules"}}
        for key in ("bad_channels", "ica_excluded", "ica_excluded_labels"):
            if key in row:
                row[key] = ";".join(map(str, row[key]))
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(paths.results / "preproc_log.csv", index=False, encoding="utf-8-sig")
    return table


def main() -> None:
    parser = cfg.add_common_args(argparse.ArgumentParser(description="CAUEEG 组别盲法预处理"))
    parser.add_argument("--serials", nargs="*", help="只处理这些编号（默认读取 preprocess_serials.txt）")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    paths = cfg.paths_from_args(args)
    paths.preprocessed.mkdir(parents=True, exist_ok=True)
    paths.preproc_logs.mkdir(parents=True, exist_ok=True)

    serials = args.serials or (paths.results / "preprocess_serials.txt").read_text(encoding="utf-8").split()
    if args.skip_existing:
        serials = [s for s in serials if not (paths.preproc_logs / f"{s}.json").exists()]
    print(f"preprocessing {len(serials)} records with {args.jobs} workers")

    done = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(_run_one, s, Path(args.caueeg_root), Path(args.project_root)): s for s in serials}
        for future in as_completed(futures):
            log = future.result()
            done += 1
            if done % 25 == 0 or not str(log.get("status")).startswith("ok"):
                print(f"[{done}/{len(serials)}] {log['serial']}: {log.get('status')}", flush=True)

    table = summarize_logs(paths)
    print(table["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
