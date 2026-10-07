"""
Step 17: Spectral measures on the 50 selected eyes-closed epochs.

Per record (channel-averaged and per channel):
- absolute band power (uV^2, 1-4/4-8/8-12/12-30 Hz) and relative power
  (denominator 1-45 Hz);
- theta/alpha power ratio;
- individual alpha frequency in 4-14 Hz on posterior channels: highest peak
  with prominence >= 10% of the local PSD range (missing if none), plus the
  4-14 Hz centre of gravity;
- specparam aperiodic offset/exponent (1-40 Hz, fixed mode) on the
  channel-averaged spectrum.

Outputs results/caueeg/spectral_global.csv, spectral_channels.csv and
spectral_psd_global.csv (channel-averaged PSD, uV^2/Hz, 1-45 Hz).
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch

import caueeg_config as cfg

POSTERIOR = ["P3", "P4", "O1", "O2", "Pz"]
TOTAL_BAND = (1.0, 45.0)
IAF_RANGE = (4.0, 14.0)


def load_epochs(npz_path) -> tuple[np.ndarray, float]:
    """Return (n_epochs, channels, samples) in uV and the sampling rate."""
    bundle = np.load(npz_path)
    data = bundle["data"].astype(np.float64)
    offsets = bundle["piece_offsets"]
    bounds = bundle["piece_bounds"]
    sfreq = float(bundle["sfreq"])
    width = int(cfg.EPOCH_SEC * sfreq)
    epochs = []
    for start, piece in zip(bundle["epoch_starts"], bundle["epoch_piece"]):
        local = int(offsets[piece] + start - bounds[piece][0])
        epochs.append(data[:, local:local + width])
    return np.stack(epochs), sfreq


def _band_sum(freqs, psd, lo, hi):
    mask = (freqs >= lo) & (freqs < hi)
    df = freqs[1] - freqs[0]
    return psd[..., mask].sum(axis=-1) * df


def alpha_peak(freqs: np.ndarray, psd: np.ndarray) -> tuple[float, float]:
    mask = (freqs >= IAF_RANGE[0]) & (freqs <= IAF_RANGE[1])
    f, p = freqs[mask], psd[mask]
    cog = float((f * p).sum() / p.sum())
    peaks, props = find_peaks(p, prominence=0.10 * (p.max() - p.min()))
    if peaks.size == 0:
        return np.nan, cog
    return float(f[peaks[np.argmax(p[peaks])]]), cog


def aperiodic_fit(freqs: np.ndarray, psd: np.ndarray) -> tuple[float, float, float]:
    from specparam import SpectralModel

    model = SpectralModel(peak_width_limits=(1.0, 8.0), max_n_peaks=6, aperiodic_mode="fixed", verbose=False)
    model.fit(freqs, psd, [1.0, 40.0])
    offset, exponent = model.get_params("aperiodic")
    r2 = model.get_metrics("gof_rsquared") if hasattr(model, "get_metrics") else np.nan
    return float(offset), float(exponent), float(np.asarray(r2).ravel()[0]) if np.size(r2) else np.nan


def spectral_record(serial: str, project_root) -> tuple[dict, list[dict], np.ndarray]:
    paths = cfg.CaueegPaths(project_root)
    epochs, sfreq = load_epochs(paths.preprocessed / f"{serial}.npz")
    freqs, psd = welch(epochs, fs=sfreq, nperseg=epochs.shape[-1], window="hann", axis=-1)
    psd = psd.mean(axis=0)  # channels x freqs, averaged over epochs
    total = _band_sum(freqs, psd, *TOTAL_BAND)

    row: dict = {"serial": serial}
    channel_rows = [{"serial": serial, "channel": ch} for ch in cfg.CHANNELS]
    global_psd = psd.mean(axis=0)
    global_total = _band_sum(freqs, global_psd, *TOTAL_BAND)
    for band, (lo, hi) in cfg.BANDS.items():
        abs_ch = _band_sum(freqs, psd, lo, hi)
        abs_global = _band_sum(freqs, global_psd, lo, hi)
        row[f"abs_{band}"] = float(abs_global)
        row[f"logabs_{band}"] = float(np.log10(abs_global))
        row[f"rel_{band}"] = float(abs_global / global_total)
        for idx, ch_row in enumerate(channel_rows):
            ch_row[f"logabs_{band}"] = float(np.log10(abs_ch[idx]))
            ch_row[f"rel_{band}"] = float(abs_ch[idx] / total[idx])
    row["theta_alpha_ratio"] = row["abs_Theta"] / row["abs_Alpha"]
    posterior = psd[[cfg.CHANNELS.index(ch) for ch in POSTERIOR]].mean(axis=0)
    row["iaf_peak"], row["iaf_cog"] = alpha_peak(freqs, posterior)
    row["ap_offset"], row["ap_exponent"], row["ap_r2"] = aperiodic_fit(freqs, global_psd)
    keep = (freqs >= TOTAL_BAND[0]) & (freqs <= TOTAL_BAND[1])
    return row, channel_rows, np.vstack([freqs[keep], global_psd[keep]])


def _job(args):
    return spectral_record(*args)


def main() -> None:
    parser = cfg.add_common_args(argparse.ArgumentParser(description="CAUEEG 频谱分析"))
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()
    paths = cfg.paths_from_args(args)
    log = pd.read_csv(paths.results / "preproc_log.csv", dtype={"serial": str})
    serials = log.loc[log["status"] == "ok", "serial"].tolist()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(_job, [(s, args.project_root) for s in serials], chunksize=8))
    pd.DataFrame([r for r, _, _ in results]).to_csv(paths.results / "spectral_global.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([row for _, rows, _ in results for row in rows]).to_csv(
        paths.results / "spectral_channels.csv", index=False, encoding="utf-8-sig")
    freqs = results[0][2][0]
    psd = pd.DataFrame(np.stack([r[2][1] for r in results]), columns=[f"{f:g}" for f in freqs])
    psd.insert(0, "serial", serials)
    psd.to_csv(paths.results / "spectral_psd_global.csv", index=False, encoding="utf-8-sig")
    print(f"spectral measures for {len(results)} records")


if __name__ == "__main__":
    main()
