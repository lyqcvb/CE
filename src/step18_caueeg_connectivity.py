"""
Step 18: Band-wise connectivity matrices (AEC, AEC-c, wPLI) for CAUEEG.

Each continuous eyes-closed piece saved by step15 is band-pass filtered
(the legacy 4th-order Butterworth, zero-phase) and Hilbert transformed as a
whole; the samples of the 50 selected epochs are then pooled. This avoids
filter and Hilbert edge effects at epoch borders.

Measures (19 x 19, diagonal 0 in the saved "graph" versions):
- aec    : |Pearson r| of amplitude envelopes (legacy LOFC definition).
- aecc   : orthogonalized AEC (Hipp et al., 2012). For each ordered pair the
           signal y is orthogonalized to x sample-wise,
           y_perp = Im(y * conj(x) / |x|), and corr(|x|, |y_perp|) is taken;
           the two directions are averaged. Negative values are set to 0.
- wpli   : debiased squared weighted phase-lag index (Vinck et al., 2011)
           from the imaginary part of the analytic cross-spectrum pooled over
           epoch samples. Negative values are set to 0.
Signed versions (aec_signed, aecc_signed, wpli_signed) keep the raw sign for
signed HOFC.

Also saved per measure: matrices from 5 consecutive 10-epoch chunks
(stability), from the first/last 25 epochs (cross-validated attack), and
HOFC convergence correlations for AEC-c with 10-50 bootstrapped epochs
(matched cohort only).
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt

import caueeg_config as cfg
from step02_connectivity import compute_signed_hofc, compute_topographical_hofc


def bandpass_sos(data: np.ndarray, band: tuple[float, float], sfreq: float, order: int = 4) -> np.ndarray:
    """Legacy 4th-order zero-phase Butterworth band-pass in SOS form (stable for 1-4 Hz at 200 Hz)."""
    sos = butter(order, band, btype="band", fs=sfreq, output="sos")
    return sosfiltfilt(sos, data, axis=1)


def band_analytic_epochs(npz_path, band: tuple[float, float]) -> np.ndarray:
    """Return analytic signals (n_epochs, channels, samples) for one band."""
    bundle = np.load(npz_path)
    data = bundle["data"].astype(np.float64)
    offsets = bundle["piece_offsets"]
    bounds = bundle["piece_bounds"]
    sfreq = float(bundle["sfreq"])
    width = int(cfg.EPOCH_SEC * sfreq)
    analytic_pieces = []
    for k in range(len(offsets) - 1):
        piece = data[:, offsets[k]:offsets[k + 1]]
        analytic_pieces.append(hilbert(bandpass_sos(piece, band, sfreq), axis=1))
    epochs = []
    for start, piece in zip(bundle["epoch_starts"], bundle["epoch_piece"]):
        local = int(start - bounds[piece][0])
        epochs.append(analytic_pieces[piece][:, local:local + width])
    return np.stack(epochs)


def _pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation between matching rows of a and b."""
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    num = (a * b).sum(axis=-1)
    den = np.sqrt((a**2).sum(axis=-1) * (b**2).sum(axis=-1))
    return np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)


def aec_signed(analytic: np.ndarray) -> np.ndarray:
    """Signed amplitude-envelope correlation; analytic is channels x samples."""
    corr = np.nan_to_num(np.corrcoef(np.abs(analytic)))
    np.fill_diagonal(corr, 0.0)
    return corr


def aecc_signed(analytic: np.ndarray) -> np.ndarray:
    """Pairwise-orthogonalized envelope correlation, averaged over both directions."""
    n = analytic.shape[0]
    env = np.abs(analytic)
    out = np.zeros((n, n))
    for i in range(n):
        x = analytic[i]
        unit = np.conj(x) / np.maximum(np.abs(x), 1e-300)
        orth = np.abs(np.imag(analytic * unit[None, :]))
        out[i] = _pearson_rows(np.broadcast_to(env[i], orth.shape), orth)
    out = 0.5 * (out + out.T)
    np.fill_diagonal(out, 0.0)
    return out


def wpli_debiased(analytic: np.ndarray) -> np.ndarray:
    """Debiased squared wPLI from pooled samples of the analytic cross-spectrum."""
    n = analytic.shape[0]
    out = np.zeros((n, n))
    for i in range(n - 1):
        im = np.imag(analytic[i][None, :] * np.conj(analytic[i + 1:]))
        total = im.sum(axis=1)
        abs_total = np.abs(im).sum(axis=1)
        sq_total = (im**2).sum(axis=1)
        den = abs_total**2 - sq_total
        value = np.where(den > 0, (total**2 - sq_total) / np.maximum(den, 1e-300), 0.0)
        out[i, i + 1:] = value
        out[i + 1:, i] = value
    return out


MEASURE_FUNCS = {"aec": aec_signed, "aecc": aecc_signed, "wpli": wpli_debiased}


def to_graph(signed: np.ndarray, measure: str) -> np.ndarray:
    """Non-negative LOFC used for networks: |r| for AEC, negatives -> 0 otherwise."""
    graph = np.abs(signed) if measure == "aec" else np.clip(signed, 0.0, None)
    np.fill_diagonal(graph, 0.0)
    return graph


def _pooled(epochs: np.ndarray) -> np.ndarray:
    return np.concatenate(list(epochs), axis=1)


def _upper(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    r = np.corrcoef(_upper(a), _upper(b))[0, 1]
    return float(r) if np.isfinite(r) else np.nan


def _for_hofc(signed: np.ndarray, measure: str, signed_hofc: bool) -> tuple[np.ndarray, np.ndarray]:
    lofc = to_graph(signed, measure)
    lofc_diag = lofc.copy()
    np.fill_diagonal(lofc_diag, 1.0)
    if signed_hofc:
        s = signed.copy()
        np.fill_diagonal(s, 1.0)
        return lofc, compute_signed_hofc(s)
    return lofc, compute_topographical_hofc(lofc_diag)


def connectivity_record(serial: str, project_root, measures: list[str], convergence: bool = True) -> list[dict]:
    paths = cfg.CaueegPaths(project_root)
    npz = paths.preprocessed / f"{serial}.npz"
    rng = np.random.default_rng(cfg.SEED + int(serial))
    convergence_rows: list[dict] = []
    for band, limits in cfg.BANDS.items():
        epochs = band_analytic_epochs(npz, limits)
        n_epochs = epochs.shape[0]
        chunks = np.array_split(np.arange(n_epochs), cfg.STABILITY_CHUNKS)
        halves = np.array_split(np.arange(n_epochs), 2)
        for measure in measures:
            func = MEASURE_FUNCS[measure]
            full = func(_pooled(epochs))
            chunk_mats = np.stack([func(_pooled(epochs[idx])) for idx in chunks])
            half_mats = np.stack([func(_pooled(epochs[idx])) for idx in halves])
            out_dir = paths.fc / measure / band
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_dir / f"{serial}.npz", full=full, chunks=chunk_mats, halves=half_mats)

            if measure != cfg.PRIMARY_MEASURE or not convergence:
                continue
            ref_lofc, ref_hofc = _for_hofc(full, measure, signed_hofc=False)
            _, ref_shofc = _for_hofc(full, measure, signed_hofc=True)
            for n_use in cfg.CONVERGENCE_EPOCHS:
                boots = 1 if n_use >= n_epochs else cfg.CONVERGENCE_BOOTSTRAPS
                for boot in range(boots):
                    idx = np.sort(rng.choice(n_epochs, size=min(n_use, n_epochs), replace=False))
                    sub = func(_pooled(epochs[idx]))
                    lofc, hofc = _for_hofc(sub, measure, signed_hofc=False)
                    _, shofc = _for_hofc(sub, measure, signed_hofc=True)
                    convergence_rows.append({
                        "serial": serial, "measure": measure, "band": band, "n_epochs": n_use, "boot": boot,
                        "r_lofc": _corr(lofc, ref_lofc), "r_hofc": _corr(hofc, ref_hofc),
                        "r_hofc_signed": _corr(shofc, ref_shofc),
                    })
    return convergence_rows


def _job(args):
    serial, root, measures, skip, convergence = args
    paths = cfg.CaueegPaths(root)
    if skip and all((paths.fc / m / b / f"{serial}.npz").exists() for m in measures for b in cfg.BANDS):
        return []
    return connectivity_record(serial, root, measures, convergence)


def main() -> None:
    parser = cfg.add_common_args(argparse.ArgumentParser(description="CAUEEG 连接矩阵 AEC / AEC-c / wPLI"))
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--measures", nargs="*", default=cfg.MEASURES)
    parser.add_argument("--serials", nargs="*")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    paths = cfg.paths_from_args(args)
    if args.serials:
        serials = args.serials
    else:
        log = pd.read_csv(paths.results / "preproc_log.csv", dtype={"serial": str})
        serials = log.loc[log["status"] == "ok", "serial"].tolist()
    # HOFC convergence (bootstrapped epoch counts) only for the primary matched cohort.
    matched_path = paths.results / "cohort_matched_111.csv"
    matched = set(pd.read_csv(matched_path, dtype={"serial": str})["serial"]) if matched_path.exists() else set()
    jobs = [(s, args.project_root, args.measures, args.skip_existing, s in matched) for s in serials]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        rows = [row for result in pool.map(_job, jobs, chunksize=4) for row in result]
    if rows:
        conv_path = paths.results / "hofc_convergence_subject.csv"
        new = pd.DataFrame(rows)
        if args.skip_existing and conv_path.exists():
            old = pd.read_csv(conv_path, dtype={"serial": str})
            new = pd.concat([old[~old["serial"].isin(new["serial"])], new], ignore_index=True)
        new.to_csv(conv_path, index=False, encoding="utf-8-sig")
    print(f"connectivity for {len(serials)} records")


if __name__ == "__main__":
    main()
