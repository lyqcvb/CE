from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from step01_eeg_io import read_eeg_set


def test_read_mat_style_set_returns_channel_by_time_data() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        set_path = Path(tmp_dir) / "sample.set"
        data = np.arange(12, dtype=np.float64).reshape(3, 4)
        sio.savemat(
            set_path,
            {
                "data": data,
                "nbchan": np.array([[3]]),
                "pnts": np.array([[4]]),
            },
        )

        loaded = read_eeg_set(str(set_path))

    np.testing.assert_allclose(loaded, data)


def test_read_hdf5_external_fdt_set() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        set_path = base / "sample.set"
        fdt_path = base / "sample.fdt"
        data = np.arange(12, dtype=np.float32).reshape(4, 3)
        data.tofile(fdt_path)

        with h5py.File(set_path, "w") as h5:
            h5.create_dataset("nbchan", data=np.array([[3]]))
            h5.create_dataset("pnts", data=np.array([[4]]))
            h5.create_dataset("datfile", data=np.array([ord(ch) for ch in "sample.fdt"]))

        loaded = read_eeg_set(str(set_path))

    np.testing.assert_allclose(loaded, data.T)


def test_hdf5_missing_fdt_reports_file_error() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        set_path = Path(tmp_dir) / "sample.set"

        with h5py.File(set_path, "w") as h5:
            h5.create_dataset("nbchan", data=np.array([[3]]))
            h5.create_dataset("pnts", data=np.array([[4]]))
            h5.create_dataset("datfile", data=np.array([ord(ch) for ch in "missing.fdt"]))

        try:
            read_eeg_set(str(set_path))
        except FileNotFoundError as exc:
            assert "Referenced .fdt file not found" in str(exc)
        else:
            raise AssertionError("missing HDF5 .fdt reference should fail clearly")
