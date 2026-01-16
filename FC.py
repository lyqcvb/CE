import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体为Times New Roman，字号为10.5（五号）
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10.5  # 五号字体约为10.5磅

# --- 1. 配置区 ---

# (!!!) 请确保此路径与您保存组平均矩阵的路径一致
DATA_PATH_ROOT = r'D:\CE\Mean' 

# 定义组、频段和网络
GROUPS = ['HC', 'MCI', 'AD']
BANDS_INFO = {
    'Delta': '0.5-4(Delta)',
    'Theta': '4-8(Theta)',
    'Alpha': '8-13(Alpha)',
    'Beta': '13-40(Beta)'
}
NETWORKS = ['lofc', 'hofc', 'ahofc']
NETWORK_TITLES = ['LOFC', 'HOFC', 'aHOFC'] # 绘图时显示的标题
BAND_NAMES = list(BANDS_INFO.keys())

MATRIX_SIZE = 16 # 每个子矩阵的大小 (16x16)

# --- 2. 辅助函数：加载矩阵 ---

def load_matrix(group, band_name, network):
    """
    根据组、频段和网络类型加载对应的平均矩阵CSV文件。
    """
    try:
        band_folder = BANDS_INFO[band_name]
        filename = f"{group}_{band_name}_{network}.csv"
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

def plot_combined_heatmaps():
    
    print("开始加载数据并绘图...")
    
    # 创建一个1行3列的图床
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
    # fig.suptitle('组平均功能连接矩阵', fontsize=20, y=0.95)
    
    mappable = None # 用于存储颜色条信息
    
    # 循环绘制 LOFC, HOFC, aHOFC
    for i, network in enumerate(NETWORKS):
        ax = axes[i]
        
        # 准备拼接超级矩阵
        super_matrix_rows = []
        
        # 1. 拼接Y轴 (HC, MCI, AD)
        for group in GROUPS:
            band_matrices = []
            
            # 2. 拼接X轴 (Delta, Theta, Alpha, Beta)
            for band in BAND_NAMES:
                matrix = load_matrix(group, band, network)
                band_matrices.append(matrix)
                
            # 将4个频段的矩阵水平拼接成一行 (16 x 64)
            group_row = np.hstack(band_matrices)
            super_matrix_rows.append(group_row)
            
        # 将3个组的行垂直拼接成最终的超级矩阵 (48 x 64)
        super_matrix = np.vstack(super_matrix_rows)
        
        # 3. 绘制热力图
        # cmap='bwr' (Blue-White-Red)
        # vmin=0, vmax=1 (0为蓝色, 0.5为白色, 1为红色)
        # 这与您的数据（相关性）和原图的外观（蓝红白）完美匹配
        mappable = ax.imshow(super_matrix, cmap='bwr', vmin=0, vmax=1, 
                            aspect='equal') # aspect='equal' 保证子块是正方形
        
        # --- 4. 设置标签 ---
        
        # 设置X轴标签 (delta, theta...) 
        # 将标签放在图的顶部，与原图一致
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top') 
        x_tick_pos = [MATRIX_SIZE * k + MATRIX_SIZE / 2 - 0.5 for k in range(len(BAND_NAMES))]
        ax.set_xticks(x_tick_pos)
        ax.set_xticklabels(BAND_NAMES, fontsize=10.5, fontfamily='Times New Roman')
        
        # 设置Y轴标签 (HC, MCI, AD)
        y_tick_pos = [MATRIX_SIZE * k + MATRIX_SIZE / 2 - 0.5 for k in range(len(GROUPS))]
        ax.set_yticks(y_tick_pos)
        ax.set_yticklabels(GROUPS, fontsize=10.5, fontfamily='Times New Roman')
        
        # 移除难看的刻度线
        ax.tick_params(top=False, bottom=False, left=False, right=False)
        
        # 设置子图标题 (LOFC, HOFC, aHOFC)
        ax.set_xlabel(NETWORK_TITLES[i], fontsize=10.5, fontfamily='Times New Roman', 
                     labelpad=20, fontweight='bold')
        
        # --- 5. 绘制分隔线 ---
        
        # 垂直线 (频段之间)
        for k in range(1, len(BAND_NAMES)):
            ax.axvline(MATRIX_SIZE * k - 0.5, color='white', linewidth=2)
            
        # 水平线 (组之间) 
        for k in range(1, len(GROUPS)):
            ax.axhline(MATRIX_SIZE * k - 0.5, color='white', linewidth=2)
            
    # --- 6. 添加共享颜色条 ---
    
    # 调整布局为颜色条腾出空间
    fig.subplots_adjust(right=0.88) 
    
    # 在图的右侧添加一个新坐标轴用于放置颜色条
    # [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.9, 0.25, 0.02, 0.5]) 
    
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    # cbar.set_label('连接强度 (0-1)', fontsize=12)
    
    print("绘图完成。")
    plt.show()

# --- 7. 运行 ---
if __name__ == "__main__":
    plot_combined_heatmaps()