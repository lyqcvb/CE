"""
Step 00: Band-pass preprocessing.

Generate band-specific .set files from 1-40 Hz EEG data before the main FC and
network-metric pipeline runs.
"""

from pathlib import Path
import argparse

import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
from tqdm import tqdm

from pipeline_config import BAND_FILTERS, BANDPASS_GROUP_NAMES, RAW_DIR

GROUPS = BANDPASS_GROUP_NAMES
RAW_DATA_ROOT = RAW_DIR

ORDER = 4

BANDS = BAND_FILTERS


def _scalar(value, default=None):
    if value is None:
        return default
    arr = np.asarray(value).squeeze()
    if arr.size == 0:
        return default
    return arr.item()


def _decode_hdf5_string(value) -> str:
    arr = np.asarray(value).flatten()
    if arr.dtype.kind in {"S", "U"}:
        return "".join(x.decode() if isinstance(x, bytes) else str(x) for x in arr).strip("\x00")
    return "".join(chr(int(x)) for x in arr if int(x) != 0)


def _as_channel_by_time(data: np.ndarray, nbchan: int | None = None) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"EEG data must be 2D, got shape {data.shape}")

    if nbchan is not None and data.shape[0] != nbchan and data.shape[1] == nbchan:
        data = data.T

    return data


def load_eeg_set(file_path: Path) -> tuple[np.ndarray, float]:
    """Load MAT-style .set or EEGLAB HDF5 .set/.fdt as channels x samples."""
    try:
        mat = sio.loadmat(str(file_path), variable_names=["data", "nbchan", "pnts", "srate"])
        if "data" not in mat:
            raise ValueError("MAT file has no 'data' variable")
        nbchan = int(_scalar(mat.get("nbchan"), mat["data"].shape[0]))
        srate = float(_scalar(mat.get("srate"), 256.0))
        return _as_channel_by_time(mat["data"], nbchan), srate
    except NotImplementedError:
        pass
    except ValueError:
        raise
    except Exception:
        pass

    import h5py

    with h5py.File(file_path, "r") as h5:
        nbchan = int(_scalar(h5.get("nbchan"), 0))
        pnts = int(_scalar(h5.get("pnts"), 0))
        srate = float(_scalar(h5.get("srate"), 256.0))

        if "data" in h5:
            data = np.array(h5["data"])
            return _as_channel_by_time(data, nbchan or None), srate

        if "datfile" not in h5:
            raise ValueError(f"{file_path} has neither 'data' nor 'datfile'")

        fdt_name = _decode_hdf5_string(h5["datfile"])

    fdt_path = file_path.with_name(fdt_name)
    if not fdt_path.exists():
        raise FileNotFoundError(f"Referenced .fdt file not found: {fdt_path}")
    if nbchan <= 0 or pnts <= 0:
        raise ValueError(f"Invalid HDF5 metadata in {file_path}: nbchan={nbchan}, pnts={pnts}")

    raw = np.fromfile(fdt_path, dtype=np.float32)
    expected = nbchan * pnts
    if raw.size != expected:
        raise ValueError(f"{fdt_path} contains {raw.size} floats, expected {expected}")

    return raw.reshape(pnts, nbchan).T.astype(np.float64), srate


def bandpass_filter(data, lowcut, highcut, fs, order=ORDER):
    nyq = 0.5 * fs
    if not 0 < lowcut < highcut < nyq:
        raise ValueError(f"Invalid band {lowcut}-{highcut} Hz for sampling rate {fs} Hz")

    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    min_samples = 3 * max(len(a), len(b)) + 1
    if data.shape[1] < min_samples:
        raise ValueError(f"Not enough samples for filtfilt: {data.shape[1]} < {min_samples}")

    return filtfilt(b, a, data, axis=1)


def process_file(file_path: Path, skip_existing: bool = False):
    base_dir = file_path.parents[1]  # data/raw/{Group} 256 100s

    if skip_existing:
        output_files = [base_dir / band_folder / file_path.name for band_folder in BANDS]
        if all(path.exists() for path in output_files):
            return

    try:
        data, srate = load_eeg_set(file_path)
    except Exception as exc:
        print(f"Error loading {file_path}: {exc}")
        return

    for band_folder, (low, high) in BANDS.items():
        out_dir = base_dir / band_folder
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / file_path.name

        if skip_existing and out_file.exists():
            continue

        try:
            filtered_data = bandpass_filter(data, low, high, srate, ORDER)
        except Exception as exc:
            print(f"Error filtering {file_path} [{band_folder}]: {exc}")
            continue

        sio.savemat(
            str(out_file),
            {
                "data": filtered_data,
                "nbchan": np.array([[filtered_data.shape[0]]], dtype=np.int32),
                "pnts": np.array([[filtered_data.shape[1]]], dtype=np.int32),
                "srate": np.array([[srate]], dtype=np.float64),
            },
        )


def main():
    parser = argparse.ArgumentParser(
        description="Apply bandpass filtering for configured EEG bands."
    )
    parser.add_argument("--group", type=str, choices=GROUPS, help="Only process one group")
    parser.add_argument("--raw-root", type=Path, default=RAW_DATA_ROOT, help="Root directory of raw EEG data")
    parser.add_argument("--source-band", default="1-40", help="Source band directory name")
    parser.add_argument("--skip-existing", action="store_true", help="Do not overwrite existing band files")
    args = parser.parse_args()

    groups_to_run = [args.group] if args.group else GROUPS

    for group in groups_to_run:
        group_dir = args.raw_root / f"{group} 256 100s"
        source_dir = group_dir / args.source_band

        if not source_dir.exists():
            print(f"Directory not found: {source_dir}. Skipping...")
            continue

        set_files = list(source_dir.glob("*.set"))
        if not set_files:
            print(f"No .set files found in {source_dir}. Skipping...")
            continue

        print(f"\nProcessing {len(set_files)} files for Group: {group}")
        for file_path in tqdm(set_files, desc=f"Filtering {group}"):
            process_file(file_path, skip_existing=args.skip_existing)

    print("\nFiltering complete.")


if __name__ == "__main__":
    main()
