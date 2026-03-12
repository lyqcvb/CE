import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# 设置全局字体为Times New Roman，字号为10.5（五号）
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20  # 五号字体约为10.5磅

# --- 1. 配置区 ---

# 请确保此路径与您保存组平均矩阵的路径一致
DATA_PATH_ROOT = BASE_DIR / 'output' / 'mean'
FIGURE_OUTPUT_DIR = BASE_DIR / 'results' / 'fc_heatmaps'

# 定义组、频段和网络
GROUPS = ['HC', 'MCI', 'AD']
BANDS_INFO = {
    'Delta': '0.5-4(Delta)',
    'Theta': '4-8(Theta)',
    'Alpha': '8-13(Alpha)',
    'Beta': '13-40(Beta)'
}
NETWORKS = ['lofc', 'hofc', 'ahofc']
BAND_NAMES = list(BANDS_INFO.keys())

MATRIX_SIZE = 16 # 每个子矩阵的大小 (16x16)

# --- 2. 辅助函数：加载矩阵 ---

def load_matrix(group, band_name, network):
    """
    根据组、频段和网络类型加载对应的平均矩阵CSV文件。
    """
    try:
        band_folder = BANDS_INFO[band_name]
        filename = f"{group}_{band_folder}_{network}.csv"
        path = os.path.join(DATA_PATH_ROOT, group, band_folder, filename)
        
        # 从CSV加载矩阵，不带表头
        matrix = pd.read_csv(path, header=None).values
        
        # 将对角线设置为空白 (NaN)，以便在热力图中显示为白色
        np.fill_diagonal(matrix, np.nan)
        
        return matrix
    
    except FileNotFoundError:
        print(f"警告: 文件未找到 {path}。将使用一个空白矩阵代替。")
        return np.full((MATRIX_SIZE, MATRIX_SIZE), np.nan)
    except Exception as e:
        print(f"加载 {path} 时出错: {e}")
        return np.full((MATRIX_SIZE, MATRIX_SIZE), np.nan)

# --- 3. 主绘图函数 ---

def build_super_matrix(network):
    """按 Group(行) × Band(列) 拼接一个网络类型的超级矩阵。"""
    super_matrix_rows = []

    for group in GROUPS:
        band_matrices = []
        for band in BAND_NAMES:
            matrix = load_matrix(group, band, network)
            band_matrices.append(matrix)

        group_row = np.hstack(band_matrices)
        super_matrix_rows.append(group_row)

    return np.vstack(super_matrix_rows)


def draw_heatmap(ax, super_matrix):
    """绘制无文字热图，并返回 mappable。"""
    mappable = ax.imshow(super_matrix, cmap='bwr', vmin=0, vmax=1, aspect='equal')

    # 移除所有文字与刻度
    ax.set_xticks([])
    ax.set_yticks([])

    # 分隔线（频段与组）
    for k in range(1, len(BAND_NAMES)):
        ax.axvline(MATRIX_SIZE * k - 0.5, color='white', linewidth=2)
    for k in range(1, len(GROUPS)):
        ax.axhline(MATRIX_SIZE * k - 0.5, color='white', linewidth=2)

    for spine in ax.spines.values():
        spine.set_visible(False)

    return mappable

def save_individual_heatmaps(output_dir=FIGURE_OUTPUT_DIR, dpi=300):
    """保存四个独立文件：3个网络热图 + 1个颜色条图。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("开始加载数据并保存图像...")

    # 保存 3 个网络热图
    for network in NETWORKS:
        super_matrix = build_super_matrix(network)

        fig, ax = plt.subplots(figsize=(8, 6))
        draw_heatmap(ax, super_matrix)

        output_file = output_dir / f"{network}_heatmap.png"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"已保存: {output_file}")

    # 保存 1 个独立颜色条图（无文字）
    fig, ax = plt.subplots(figsize=(1.2, 6))
    norm = plt.Normalize(vmin=0, vmax=1)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap='bwr')
    mappable.set_array([])

    cbar = fig.colorbar(mappable, cax=ax)
    cbar.set_ticks([])
    cbar.outline.set_visible(False)
    ax.tick_params(left=False, right=False, labelleft=False, labelright=False)

    colorbar_file = output_dir / 'colorbar.png'
    fig.savefig(colorbar_file, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"已保存: {colorbar_file}")

    print("全部图像保存完成。")

# --- 7. 运行 ---
if __name__ == "__main__":
    save_individual_heatmaps()