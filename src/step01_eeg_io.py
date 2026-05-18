"""
Step 01: EEG data loading.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio


def _scalar(value: object, default: object = None) -> object:
    if value is None:
        return default
    arr = np.asarray(value).squeeze()
    if arr.size == 0:
        return default
    return arr.item()


def _decode_hdf5_string(value: object) -> str:
    arr = np.asarray(value).flatten()
    if arr.dtype.kind in {"S", "U"}:
        return "".join(x.decode() if isinstance(x, bytes) else str(x) for x in arr).strip("\x00")
    return "".join(chr(int(x)) for x in arr if int(x) != 0)


def _as_channel_by_time(data: np.ndarray, nbchan: int | None = None) -> np.ndarray:
    eeg_data = np.asarray(data, dtype=np.float64)
    if eeg_data.ndim != 2:
        raise ValueError(f"EEG data must be 2D, got shape {eeg_data.shape}")

    if nbchan is not None and eeg_data.shape[0] != nbchan and eeg_data.shape[1] == nbchan:
        eeg_data = eeg_data.T

    return eeg_data


def _load_hdf5_eeg_set(set_path: Path) -> tuple[np.ndarray, int]:
    with h5py.File(set_path, "r") as h5:
        nbchan_raw = _scalar(h5.get("nbchan"), None)
        nbchan = int(nbchan_raw) if nbchan_raw is not None else None

        if "data" in h5:
            eeg_data = _as_channel_by_time(np.array(h5["data"]), nbchan)
            return eeg_data, int(eeg_data.shape[1])

        pnts_raw = _scalar(h5.get("pnts"), None)
        if nbchan is None or pnts_raw is None:
            raise ValueError(f"{set_path} is missing required HDF5 metadata: nbchan/pnts")
        pnts = int(pnts_raw)
        if nbchan <= 0 or pnts <= 0:
            raise ValueError(f"Invalid HDF5 metadata in {set_path}: nbchan={nbchan}, pnts={pnts}")

        if "datfile" not in h5:
            raise ValueError(f"{set_path} has neither embedded 'data' nor external 'datfile'")

        fdt_name = _decode_hdf5_string(h5["datfile"])

    fdt_path = set_path.with_name(fdt_name)
    if not fdt_path.exists():
        raise FileNotFoundError(f"Referenced .fdt file not found: {fdt_path}")

    raw = np.fromfile(fdt_path, dtype=np.float32)
    expected = nbchan * pnts
    if raw.size != expected:
        raise ValueError(f"{fdt_path} contains {raw.size} floats, expected {expected}")

    return raw.reshape(pnts, nbchan).T.astype(np.float64), pnts


def _load_mat_eeg_set(set_path: Path) -> tuple[np.ndarray, int]:
    mat_data = sio.loadmat(str(set_path), variable_names=["data", "nbchan", "pnts"])
    if "data" not in mat_data:
        raise ValueError(f"{set_path} has no 'data' variable")

    nbchan = int(_scalar(mat_data.get("nbchan"), mat_data["data"].shape[0]))
    eeg_data = _as_channel_by_time(mat_data["data"], nbchan)
    pnts = int(_scalar(mat_data.get("pnts"), eeg_data.shape[1]))
    return eeg_data, pnts


def read_eeg_set(set_path: str, n_samples: int = 25601) -> np.ndarray:
    """
    Read EEGLAB .set/.fdt data as channels x samples.

    HDF5-backed EEGLAB files are parsed first. Only a low-level HDF5 open
    failure falls back to scipy.io.loadmat for MAT-style .set files; malformed
    HDF5 metadata or missing .fdt files are reported directly.
    """
    path = Path(set_path).resolve()

    if h5py.is_hdf5(path):
        eeg_data, pnts = _load_hdf5_eeg_set(path)
    else:
        eeg_data, pnts = _load_mat_eeg_set(path)

    n_use = min(n_samples, pnts, eeg_data.shape[1])
    return eeg_data[:, :n_use].astype(np.float64)
