"""Step 12: FC heatmap visualization for the current LOFC/HOFC pipeline."""

from __future__ import annotations

from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_config import (
    BAND_FOLDERS_BY_LABEL,
    BANDS,
    FC_TYPES_LOWER,
    GROUP_NAMES,
    RESULTS_DIR,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH_ROOT = BASE_DIR / "output" / "mean"
FIGURE_OUTPUT_DIR = RESULTS_DIR / "fc_heatmaps"

GROUPS = GROUP_NAMES
BANDS_INFO = BAND_FOLDERS_BY_LABEL
NETWORKS = FC_TYPES_LOWER
BAND_NAMES = BANDS
MATRIX_SIZE = 16
NETWORK_LABELS = {network: network.upper() for network in NETWORKS}
COLOR_TICKS = [0.0, 0.5, 1.0]


def setup_style() -> None:
    """Apply journal-style typography for heatmap outputs."""
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "font.size": 9.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 9.0,
            "axes.unicode_minus": False,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
            "svg.fonttype": "none",
        }
    )


def get_heatmap_cmap():
    cmap = plt.get_cmap("bwr").copy()
    cmap.set_bad("white")
    return cmap


def load_matrix(group: str, band_name: str, network: str) -> np.ndarray:
    """Load one group-average FC matrix."""
    try:
        band_folder = BANDS_INFO[band_name]
        filename = f"{group}_{band_folder}_{network}.csv"
        path = os.path.join(DATA_PATH_ROOT, group, band_folder, filename)
        matrix = pd.read_csv(path, header=None).values
        np.fill_diagonal(matrix, np.nan)
        return matrix
    except FileNotFoundError:
        print(f"警告: 文件未找到 {path}。将使用一个空白矩阵代替。")
        return np.full((MATRIX_SIZE, MATRIX_SIZE), np.nan)
    except Exception as exc:
        print(f"加载 {path} 时出错: {exc}")
        return np.full((MATRIX_SIZE, MATRIX_SIZE), np.nan)


def build_super_matrix(network: str) -> np.ndarray:
    """Concatenate one network type as Group(row) by Band(column)."""
    super_matrix_rows = []
    for group in GROUPS:
        band_matrices = [load_matrix(group, band, network) for band in BAND_NAMES]
        super_matrix_rows.append(np.hstack(band_matrices))
    return np.vstack(super_matrix_rows)


def draw_annotated_heatmap(
    ax: plt.Axes,
    super_matrix: np.ndarray,
    network: str,
    show_group_labels: bool = True,
):
    """Draw one network panel with group and frequency annotations."""
    mappable = ax.imshow(super_matrix, cmap=get_heatmap_cmap(), vmin=0, vmax=1, aspect="equal")

    band_centers = [MATRIX_SIZE * (idx + 0.5) - 0.5 for idx in range(len(BAND_NAMES))]
    group_centers = [MATRIX_SIZE * (idx + 0.5) - 0.5 for idx in range(len(GROUPS))]

    ax.set_title(NETWORK_LABELS.get(network, network.upper()), fontweight="bold", pad=22)
    ax.set_xticks(band_centers)
    ax.set_xticklabels(BAND_NAMES)
    ax.xaxis.tick_top()
    ax.tick_params(
        axis="x",
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False,
        length=0,
        pad=2,
    )

    ax.set_yticks(group_centers)
    ax.set_yticklabels(GROUPS if show_group_labels else [])
    ax.tick_params(axis="y", left=False, right=False, length=0, pad=4)

    for idx in range(1, len(BAND_NAMES)):
        ax.axvline(MATRIX_SIZE * idx - 0.5, color="white", linewidth=1.6)
    for idx in range(1, len(GROUPS)):
        ax.axhline(MATRIX_SIZE * idx - 0.5, color="white", linewidth=1.6)

    for spine in ax.spines.values():
        spine.set_visible(False)

    return mappable


def add_colorbar(fig: plt.Figure, cax: plt.Axes, mappable) -> None:
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_ticks(COLOR_TICKS)
    cbar.set_ticklabels(["0", "0.5", "1"])
    cbar.set_label("Connection strength", rotation=270, labelpad=12)
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(length=2, width=0.6, pad=2)


def save_combined_heatmap(
    output_dir: Path = FIGURE_OUTPUT_DIR,
    dpi: int = 450,
    output_stem: str = "lofc_hofc_heatmap",
) -> None:
    """Save a single LOFC/HOFC overview panel with a shared colorbar."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.4, 3.15))
    gridspec = fig.add_gridspec(
        1,
        len(NETWORKS) + 1,
        width_ratios=[1.0] * len(NETWORKS) + [0.07],
        wspace=0.08,
    )

    mappable = None
    for idx, network in enumerate(NETWORKS):
        ax = fig.add_subplot(gridspec[0, idx])
        super_matrix = build_super_matrix(network)
        mappable = draw_annotated_heatmap(ax, super_matrix, network, show_group_labels=(idx == 0))

    cax = fig.add_subplot(gridspec[0, -1])
    add_colorbar(fig, cax, mappable)

    output_png = output_dir / f"{output_stem}.png"
    output_svg = output_dir / f"{output_stem}.svg"
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output_svg, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"已保存: {output_png}")
    print(f"已保存: {output_svg}")


def save_individual_heatmaps(output_dir: Path = FIGURE_OUTPUT_DIR, dpi: int = 450) -> None:
    """Save annotated heatmaps for each network plus a standalone colorbar."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for network in NETWORKS:
        fig = plt.figure(figsize=(4.4, 3.15))
        gridspec = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.07], wspace=0.08)

        ax = fig.add_subplot(gridspec[0, 0])
        super_matrix = build_super_matrix(network)
        mappable = draw_annotated_heatmap(ax, super_matrix, network, show_group_labels=True)

        cax = fig.add_subplot(gridspec[0, 1])
        add_colorbar(fig, cax, mappable)

        output_png = output_dir / f"{network}_heatmap.png"
        output_svg = output_dir / f"{network}_heatmap.svg"
        fig.savefig(output_png, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
        fig.savefig(output_svg, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        print(f"已保存: {output_png}")
        print(f"已保存: {output_svg}")

    fig, ax = plt.subplots(figsize=(1.2, 6))
    norm = plt.Normalize(vmin=0, vmax=1)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=get_heatmap_cmap())
    mappable.set_array([])
    add_colorbar(fig, ax, mappable)

    colorbar_file = output_dir / "colorbar.png"
    fig.savefig(colorbar_file, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"已保存: {colorbar_file}")


def save_heatmaps(output_dir: Path = FIGURE_OUTPUT_DIR, dpi: int = 450) -> None:
    setup_style()
    save_combined_heatmap(output_dir=output_dir, dpi=dpi)
    save_individual_heatmaps(output_dir=output_dir, dpi=dpi)


if __name__ == "__main__":
    save_heatmaps()
