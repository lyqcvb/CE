"""
Shared constants for the CE EEG analysis pipeline.

These values define the public folder and label conventions used across the
batch pipeline, post-processing, statistical analysis, and visualization steps.
"""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "output"
RESULTS_DIR = BASE_DIR / "results"

GROUP_DIRS = {
    "HC": "HC 256 100s",
    "MCI": "MCI 256 100s",
    "AD": "AD 256 100s",
}
GROUP_NAMES = list(GROUP_DIRS.keys())
BANDPASS_GROUP_NAMES = ["AD", "HC", "MCI"]

BAND_LABELS = {
    "1-4(Delta)": "Delta",
    "4-8(Theta)": "Theta",
    "8-12(Alpha)": "Alpha",
    "12-30(Beta)": "Beta",
    "30-40(Gamma)": "Gamma",
}
BANDS = list(BAND_LABELS.values())
BAND_FOLDERS_BY_LABEL = {label: folder for folder, label in BAND_LABELS.items()}
BAND_FILTERS = {
    "1-4(Delta)": (1.0, 4.0),
    "4-8(Theta)": (4.0, 8.0),
    "8-12(Alpha)": (8.0, 12.0),
    "12-30(Beta)": (12.0, 30.0),
    "30-40(Gamma)": (30.0, 40.0),
}

FC_TYPES_LOWER = ["lofc", "hofc", "ahofc"]
FC_TYPES_UPPER = ["LOFC", "HOFC", "AHOFC"]

THRESHOLD_DEFAULT = 0.35
THRESHOLD_RICHCLUB = THRESHOLD_DEFAULT

# Fixed-density sensitivity analysis. Each graph keeps the strongest
# off-diagonal connections at these proportions, then scalar graph metrics are
# integrated with a normalized trapezoid AUC.
SPARSITY_LEVELS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

# Main hub definition uses a fixed top ratio so every 16-node network has a
# comparable number of hubs. The legacy mean+std rule is still saved as a
# strict supplementary label.
HUB_TOP_RATIO = 0.20
REPRESENTATIVE_CORE_THRESHOLDS = [0.20, 0.30, 0.40]

# Random attack is averaged across repeats. Keep the default moderate because
# the full pipeline already runs rich-club randomizations for every subject.
RANDOM_ATTACK_REPEATS = 1000

# Stable core nodes must be selected in at least this many density thresholds.
STABLE_CORE_MIN_DENSITIES = 4

# Epoch-level FC stability defaults. The batch pipeline can override this from
# the CLI; epochs are contiguous equal-length chunks of the already loaded data.
FC_STABILITY_EPOCHS = 5
FC_STABILITY_MIN_SAMPLES = 32
FC_STABILITY_CV_EPS = 1e-12

DEFAULT_CHANNEL_NAMES = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
]
