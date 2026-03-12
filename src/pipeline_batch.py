"""
pipeline_batch.py
=================
批量处理脚本：遍历所有组别（HC/MCI/AD）× 频段 × 被试，
从原始 EEG .set 文件计算 FC 矩阵和所有网络指标，并将结果写入对应目录。

使用方法
--------
    # 处理全部数据
    uv run python src/pipeline_batch.py

    # 仅处理指定组和频段（测试用）
    uv run python src/pipeline_batch.py --group HC --band "0.5-4(Delta)"

    # 仅生成 FC 矩阵（不计算网络指标）
    uv run python src/pipeline_batch.py --fc-only

输出目录与文件说明
------------------
计算结果写入以下目录（Python 计算结果与原始数据分离）：
  output/fc_matrices/{fc_type}/{group}/{band}/{subject}.csv   — FC 矩阵（16×16）
  output/core_nodes/{fc_type}/{group}/{band}/{subject}_core_nodes.csv
  output/rich_club/{fc_type}/{group}/{band}/{subject}.csv     — Rich-club 向量
  output/target_attack/{fc_type}/{group}/{band}/{subject}.csv — 靶向攻击向量
  output/mean/{group}/{band}/                                 — 组均值矩阵（3 个 FC 类型）
  output/all_scalar_metrics_summary.csv                       — 全部被试标量指标汇总
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# 抑制常量通道（方差=0）引发的 Pearson 相关 RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, os.path.dirname(__file__))
import brain_metrics as bm

# ─────────────────────────────────────────────────────────
# 路径与参数配置
# ─────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
RAW_DIR    = BASE_DIR / "data" / "raw"
OUT_FC     = BASE_DIR / "output" / "fc_matrices"
OUT_CORE   = BASE_DIR / "output" / "core_nodes"
OUT_RICH   = BASE_DIR / "output" / "rich_club"
OUT_ATTACK = BASE_DIR / "output" / "target_attack"
OUT_MEAN   = BASE_DIR / "output" / "mean"
SUMMARY_CSV = BASE_DIR / "output" / "all_scalar_metrics_summary.csv"

# 组/目录名 映射
GROUPS = {
    "HC":  "HC 256 100s",
    "MCI": "MCI 256 100s",
    "AD":  "AD 256 100s",
}

# 频段/目录名 映射（目录名 → Band label for CSV）
BAND_LABELS = {
    "0.5-4(Delta)": "Delta",
    "4-8(Theta)":   "Theta",
    "8-13(Alpha)":  "Alpha",
    "13-40(Beta)":  "Beta",
}

# FC 类型
FC_TYPES = ["lofc", "hofc", "ahofc"]

# 二值化阈值
THRESHOLD_DEFAULT  = 0.35   # 一般指标
THRESHOLD_RICHCLUB = 0.40   # 富人俱乐部

# Rich-club / target attack 参数
KLEVEL_LOFC_HOFC = 16
KLEVEL_AHOFC     = 32
N_RAND           = 1000


# ─────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────

def save_csv(arr: np.ndarray, path: Path, header: bool = False) -> None:
    """保存 ndarray 为 CSV（无表头）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(arr).to_csv(path, index=False, header=header)


def list_subjects(raw_group_band_dir: Path) -> list[str]:
    """返回目录下所有 .set 文件名（含扩展名）。"""
    if not raw_group_band_dir.exists():
        return []
    return sorted(f.name for f in raw_group_band_dir.iterdir() if f.suffix == ".set")


# ─────────────────────────────────────────────────────────
# 主计算流程
# ─────────────────────────────────────────────────────────

def process_subject(
    set_path: Path,
    group: str,
    band_dir: str,
    band_label: str,
    fc_only: bool = False,
    n_rand: int = N_RAND,
    seed: int = 42,
) -> dict | None:
    """
    处理单个被试，计算 FC 矩阵并（可选）计算指标，保存结果。

    Parameters
    ----------
    set_path  : .set 文件完整路径
    group     : "HC" / "MCI" / "AD"
    band_dir  : 频段目录名，如 "13-40(Beta)"
    band_label: CSV 中的频段标签，如 "Beta"
    fc_only   : 若为 True，仅保存 FC 矩阵，跳过网络指标计算

    Returns
    -------
    scalar_row : dict 或 None（出错时）
    """
    subj_name = set_path.stem  # 去掉 .set 后缀

    try:
        # ── 1. 读取 EEG ──────────────────────────────────────────
        eeg   = bm.read_eeg_set(str(set_path))
        lofc  = bm.compute_lofc(eeg)
        hofc  = bm.compute_hofc(lofc)
        ahofc = bm.compute_ahofc(lofc, hofc)

        # ── 2. 保存 FC 矩阵 CSV ─────────────────────────────────
        for fc_name, mat in [("lofc", lofc), ("hofc", hofc), ("ahofc", ahofc)]:
            out_path = OUT_FC / fc_name / group / band_dir / f"{subj_name}.csv"
            save_csv(mat, out_path)

        if fc_only:
            return {"Group": group, "Band": band_label, "SubjectID": set_path.name}

        # ── 3. 标量指标（MeanConn / Eglob / Cluster）─────────────
        scalar = bm.compute_all_scalar_metrics(lofc, hofc, ahofc)
        scalar_row = {
            "Group": group,
            "Band": band_label,
            "SubjectID": set_path.name,
            **scalar,
        }

        # ── 4. Hub 节点识别 & 核心节点 CSV ─────────────────────
        for fc_name, mat, is_dir in [
            ("lofc",  lofc,  False),
            ("hofc",  hofc,  False),
            ("ahofc", ahofc, True),
        ]:
            G = bm.binarize(mat, THRESHOLD_DEFAULT)
            is_hub, _ = bm.identify_hub_nodes(G, directed=is_dir)
            core_rows = [{"NodeIndex": i, "IsCoreNode": int(is_hub[i])}
                         for i in range(len(is_hub))]
            out_path = OUT_CORE / fc_name / group / band_dir / f"{subj_name}_core_nodes.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(core_rows).to_csv(out_path, index=False)

        # ── 5. Rich-club 系数 ───────────────────────────────────
        G_lofc_rc  = bm.binarize(lofc,  THRESHOLD_RICHCLUB)
        G_hofc_rc  = bm.binarize(hofc,  THRESHOLD_RICHCLUB)
        G_ahofc_rc = bm.binarize(ahofc, THRESHOLD_RICHCLUB)

        rc_lofc  = bm.compute_rich_club(G_lofc_rc,  directed=False, klevel=KLEVEL_LOFC_HOFC, n_rand=n_rand, seed=seed)
        rc_hofc  = bm.compute_rich_club(G_hofc_rc,  directed=False, klevel=KLEVEL_LOFC_HOFC, n_rand=n_rand, seed=seed)
        rc_ahofc = bm.compute_rich_club(G_ahofc_rc, directed=True,  klevel=KLEVEL_AHOFC,     n_rand=n_rand, seed=seed)

        for fc_name, rc in [("lofc", rc_lofc), ("hofc", rc_hofc), ("ahofc", rc_ahofc)]:
            out_path = OUT_RICH / fc_name / group / band_dir / f"{subj_name}.csv"
            save_csv(rc.reshape(1, -1), out_path)

        # ── 6. 靶向攻击 ─────────────────────────────────────────
        G_lofc_b  = bm.binarize(lofc,  THRESHOLD_DEFAULT)
        G_hofc_b  = bm.binarize(hofc,  THRESHOLD_DEFAULT)
        G_ahofc_b = bm.binarize(ahofc, THRESHOLD_DEFAULT)

        ta_lofc  = bm.compute_target_attack(G_lofc_b,  directed=False)
        ta_hofc  = bm.compute_target_attack(G_hofc_b,  directed=False)
        ta_ahofc = bm.compute_target_attack(G_ahofc_b, directed=True)

        for fc_name, ta in [("lofc", ta_lofc), ("hofc", ta_hofc), ("ahofc", ta_ahofc)]:
            out_path = OUT_ATTACK / fc_name / group / band_dir / f"{subj_name}.csv"
            save_csv(ta.reshape(1, -1), out_path)

        return scalar_row

    except Exception as exc:
        print(f"  [!] 错误 {subj_name}: {exc}")
        traceback.print_exc()
        return None


def compute_group_mean(
    group: str, band_dir: str
) -> None:
    """计算并保存指定组/频段的组均值 FC 矩阵。"""
    for fc_name in FC_TYPES:
        fc_dir = OUT_FC / fc_name / group / band_dir
        if not fc_dir.exists():
            continue
        mats = []
        for csv_file in sorted(fc_dir.glob("*.csv")):
            try:
                m = pd.read_csv(csv_file, header=None).values
                if m.shape == (16, 16):
                    mats.append(m)
            except Exception:
                pass
        if not mats:
            continue
        mean_mat = np.mean(mats, axis=0)
        # 存储到 output/mean/{group}/{band_dir}/{group}_{band_label}_{fc_name}.csv
        out_path = OUT_MEAN / group / band_dir / f"{group}_{band_dir}_{fc_name}.csv"
        save_csv(mean_mat, out_path)


# ─────────────────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="批量计算脑网络 FC 矩阵和指标")
    parser.add_argument("--group", choices=list(GROUPS.keys()), default=None,
                        help="仅处理指定组（默认：处理全部）")
    parser.add_argument("--band",  default=None,
                        help="仅处理指定频段目录名，如 '13-40(Beta)'（默认：全部）")
    parser.add_argument("--fc-only", action="store_true",
                        help="仅保存 FC 矩阵，跳过网络指标、富人俱乐部等耗时计算")
    parser.add_argument("--n-rand", type=int, default=N_RAND,
                        help="富人俱乐部随机化次数（默认 1000）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--skip-existing", action="store_true",
                        help="跳过已有 FC 矩阵的被试（增量处理）")
    args = parser.parse_args()

    groups_to_run = {args.group: GROUPS[args.group]} if args.group else GROUPS
    bands_to_run  = {args.band: BAND_LABELS[args.band]} if args.band else BAND_LABELS

    all_scalar_rows: list[dict] = []
    total_subjects  = 0
    total_done      = 0
    total_errors    = 0

    for group, group_dir in groups_to_run.items():
        for band_dir, band_label in bands_to_run.items():
            raw_dir = RAW_DIR / group_dir / band_dir
            subjects = list_subjects(raw_dir)
            if not subjects:
                print(f"[跳过] 目录不存在或无 .set 文件: {raw_dir}")
                continue

            print(f"\n{'='*60}")
            print(f"处理 {group} / {band_label}  ({len(subjects)} 被试)")
            print(f"{'='*60}")
            total_subjects += len(subjects)

            for i, subj_file in enumerate(subjects, 1):
                set_path = raw_dir / subj_file
                subj_name = set_path.stem

                # 增量模式：跳过已有 FC 矩阵的被试
                if args.skip_existing:
                    existing = OUT_FC / "lofc" / group / band_dir / f"{subj_name}.csv"
                    if existing.exists():
                        print(f"  [{i:3d}/{len(subjects)}] 跳过（已存在）: {subj_name}")
                        continue

                t0 = time.time()
                print(f"  [{i:3d}/{len(subjects)}] {subj_name} ... ", end="", flush=True)

                row = process_subject(
                    set_path=set_path,
                    group=group,
                    band_dir=band_dir,
                    band_label=band_label,
                    fc_only=args.fc_only,
                    n_rand=args.n_rand,
                    seed=args.seed,
                )

                elapsed = time.time() - t0
                if row is not None:
                    total_done += 1
                    all_scalar_rows.append(row)
                    print(f"✓ ({elapsed:.1f}s)")
                else:
                    total_errors += 1
                    print(f"✗ 失败")

            # 计算组均值矩阵
            print(f"  计算 {group}/{band_dir} 组均值矩阵 ...", end=" ", flush=True)
            compute_group_mean(group, band_dir)
            print("✓")

    # ── 汇总标量指标 CSV ──────────────────────────────────
    if all_scalar_rows and not args.fc_only:
        df_new = pd.DataFrame(all_scalar_rows)
        if SUMMARY_CSV.exists():
            df_old = pd.read_csv(SUMMARY_CSV)
            # 按 Group/Band/SubjectID 去重，以新结果为准
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(
                subset=["Group", "Band", "SubjectID"], keep="last"
            )
        else:
            df_combined = df_new

        df_combined.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
        print(f"\n已更新汇总指标: {SUMMARY_CSV}  ({len(df_combined)} 行)")

    # ── 汇总输出 ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"批量处理完成")
    print(f"  总被试: {total_subjects}")
    print(f"  成功:   {total_done}")
    print(f"  失败:   {total_errors}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
