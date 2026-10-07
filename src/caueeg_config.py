"""
Shared constants for the CAUEEG replication pipeline (step14-step21).

The CAUEEG dataset is read-only. Generated files go to
``{project_root}/{data,output,results}/caueeg`` which are gitignored, so the
legacy 16-channel results under data/raw, output/ and results/ are untouched.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

CAUEEG_ROOT = Path(os.environ.get("CAUEEG_ROOT", r"F:\数据\caueeg-dataset"))
DEFAULT_PROJECT_ROOT = Path(os.environ.get("CE_PROJECT_ROOT", r"E:\Paper_code\CE"))

SFREQ = 200.0

# EDF order of the 19 EEG channels after stripping the "-AVG" suffix.
EDF_CHANNELS = [
    "Fp1", "F3", "C3", "P3", "O1", "Fp2", "F4", "C4", "P4", "O2",
    "F7", "T3", "T5", "F8", "T4", "T6", "FZ", "CZ", "PZ",
]
# MNE standard_1020 spelling (Fz/Cz/Pz); T3/T4/T5/T6 exist in that montage.
CHANNELS = [name if not name.endswith("Z") else name[0] + "z" for name in EDF_CHANNELS]
N_CHANNELS = len(CHANNELS)

GROUPS = ["HC", "MCI", "AD"]  # MCI = amnestic MCI in this dataset
GROUP_DISPLAY = {"HC": "HC", "MCI": "aMCI", "AD": "AD"}

BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 12.0),
    "Beta": (12.0, 30.0),
}
BAND_NAMES = list(BANDS)

# Pre-specified preprocessing parameters (docs/caueeg_analysis_plan.md).
AGE_RANGE = (55, 90)
LINE_FREQ = 60.0
ANALYSIS_BAND = (1.0, 45.0)
# ICLabel expects 1-100 Hz; at 200 Hz the Nyquist is 100 Hz, so only high-pass.
ICA_FIT_HIGHPASS = 1.0
ICA_REJECT_UV = 500.0
ICLABEL_CLASSES = ["Brain", "Muscle", "Eye", "Heart", "Line Noise", "Channel Noise", "Other"]
ICLABEL_REJECT = ("Muscle", "Eye", "Heart", "Line Noise", "Channel Noise")
ICLABEL_THRESHOLD = 0.80
EPOCH_SEC = 2.0
EPOCH_EDGE_SEC = 1.0
EPOCH_PTP_UV = 150.0
N_EPOCHS = 50
HV_BUFFER_SEC = 30.0
EVENT_ARTIFACT_PAD_SEC = 1.0
RECORD_TAIL_SEC = 5.0
MIN_DELTA_SPAN_SEC = 6.0
BAD_FLAT_SD_UV = 0.5
BAD_LOGVAR_Z = 5.0
BAD_NEIGHBOR_CORR = 0.4
BAD_NEIGHBORS = 4
MAX_BAD_CHANNELS = 3
SPAN_MARGIN_SEC = 10.0

MEASURES = ["aec", "aecc", "wpli"]
PRIMARY_MEASURE = "aecc"
DENSITIES = [0.20, 0.25, 0.30, 0.35, 0.40]
THRESHOLD_SENSITIVITY = 0.35
RANDOM_ATTACK_REPEATS = 100
NULL_REPEATS = 50
STABILITY_CHUNKS = 5
CONVERGENCE_EPOCHS = [10, 20, 30, 40, 50]
CONVERGENCE_BOOTSTRAPS = 10
SEED = 42


@dataclass(frozen=True)
class CaueegPaths:
    project_root: Path

    @property
    def data(self) -> Path:
        return self.project_root / "data" / "caueeg"

    @property
    def output(self) -> Path:
        return self.project_root / "output" / "caueeg"

    @property
    def results(self) -> Path:
        return self.project_root / "results" / "caueeg"

    @property
    def events(self) -> Path:
        return self.data / "events_corrected"

    @property
    def preprocessed(self) -> Path:
        return self.data / "preprocessed"

    @property
    def preproc_logs(self) -> Path:
        return self.results / "preproc_logs"

    @property
    def fc(self) -> Path:
        return self.output / "fc"

    @property
    def figures(self) -> Path:
        return self.results / "figures"

    def ensure(self) -> "CaueegPaths":
        for path in (self.data, self.output, self.results):
            path.mkdir(parents=True, exist_ok=True)
        return self


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT,
                        help="根目录：输出写入 {root}/{data,output,results}/caueeg")
    parser.add_argument("--caueeg-root", type=Path, default=CAUEEG_ROOT,
                        help="CAUEEG 数据集根目录（只读）")
    return parser


def paths_from_args(args: argparse.Namespace) -> CaueegPaths:
    return CaueegPaths(Path(args.project_root)).ensure()
