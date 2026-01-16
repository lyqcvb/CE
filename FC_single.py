import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体为Times New Roman，字号为10.5（五号）
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10.5  # 五号字体约为10.5磅

# --- 1. 配置区 ---

# 数据路径
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
NETWORK_TITLES = ['LOFC', 'HOFC', 'aHOFC']

MATRIX_SIZE = 16  # 每个矩阵的大小 (16x16)

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
        print(f"错误: 文件未找到 {path}")
        return None
    except Exception as e:
        print(f"加载 {path} 时出错: {e}")
        return None

# --- 3. 绘制单个矩阵的函数 ---

def plot_single_matrix(group, band_name, network, save_fig=False, output_dir='output', 
                      minimal_mode=False):
    """
    绘制单个组别、单个频段、单个网络类型的功能连接矩阵
    
    参数:
        group: 组别 ('HC', 'MCI', 'AD')
        band_name: 频段名称 ('Delta', 'Theta', 'Alpha', 'Beta')
        network: 网络类型 ('lofc', 'hofc', 'ahofc')
        save_fig: 是否保存图片 (默认False)
        output_dir: 保存图片的目录 (默认'output')
        minimal_mode: 仅保留主图，不显示刻度、标签、标题、颜色条等 (默认False)
    """
    
    print(f"正在绘制: {group} - {band_name} - {network.upper()}...")
    
    # 加载矩阵
    matrix = load_matrix(group, band_name, network)
    
    if matrix is None:
        print("无法加载矩阵，绘图终止。")
        return
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # 绘制热力图
    mappable = ax.imshow(matrix, cmap='bwr', vmin=0, vmax=1, aspect='equal')
    
    if minimal_mode:
        # 最小化模式：移除所有装饰元素
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        # 移除边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        # 移除坐标轴
        ax.axis('off')
        
        # 调整布局使图片填充整个区域，无边距
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.margins(0, 0)
        ax.set_xlim(-0.5, MATRIX_SIZE - 0.5)
        ax.set_ylim(MATRIX_SIZE - 0.5, -0.5)
        
    else:
        # 正常模式：显示所有装饰元素
        # 设置标题
        network_title = NETWORK_TITLES[NETWORKS.index(network)]
        title = f"{group} - {band_name} - {network_title}"
        ax.set_title(title, fontsize=12, fontfamily='Times New Roman', fontweight='bold', pad=15)
        
        # 设置刻度标签（节点编号）
        tick_positions = range(MATRIX_SIZE)
        tick_labels = [str(i+1) for i in tick_positions]
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=9, fontfamily='Times New Roman')
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontsize=9, fontfamily='Times New Roman')
        
        # 设置轴标签
        ax.set_xlabel('Node', fontsize=10.5, fontfamily='Times New Roman', fontweight='bold')
        ax.set_ylabel('Node', fontsize=10.5, fontfamily='Times New Roman', fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Connection Strength', fontsize=10.5, fontfamily='Times New Roman')
        cbar.ax.tick_params(labelsize=9)
        
        # 调整布局
        plt.tight_layout()
    
    # 保存图片
    if save_fig:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        suffix = '_minimal' if minimal_mode else ''
        filename = f"{group}_{band_name}_{network}{suffix}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0 if minimal_mode else 0.1)
        print(f"图片已保存至: {filepath}")
    
    print("绘图完成。")
    plt.show()

# --- 4. 绘制所有三个网络类型的单个组别和频段 ---

def plot_single_band_group_all_networks(group, band_name, save_fig=False, output_dir='output',
                                       minimal_mode=False):
    """
    绘制单个组别、单个频段的所有三个网络类型（LOFC, HOFC, aHOFC）
    
    参数:
        group: 组别 ('HC', 'MCI', 'AD')
        band_name: 频段名称 ('Delta', 'Theta', 'Alpha', 'Beta')
        save_fig: 是否保存图片 (默认False)
        output_dir: 保存图片的目录 (默认'output')
        minimal_mode: 仅保留主图，不显示刻度、标签、标题、颜色条等 (默认False)
    """
    
    print(f"正在绘制: {group} - {band_name} - 所有网络...")
    
    # 创建1行3列的子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    mappable = None
    
    # 循环绘制三个网络
    for i, network in enumerate(NETWORKS):
        ax = axes[i]
        
        # 加载矩阵
        matrix = load_matrix(group, band_name, network)
        
        if matrix is None:
            print(f"警告: 无法加载 {network} 的矩阵，跳过。")
            continue
        
        # 绘制热力图
        mappable = ax.imshow(matrix, cmap='bwr', vmin=0, vmax=1, aspect='equal')
        
        if minimal_mode:
            # 最小化模式：移除所有装饰元素
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title('')
            # 移除边框和坐标轴
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.axis('off')
            
            # 设置精确的显示范围
            ax.set_xlim(-0.5, MATRIX_SIZE - 0.5)
            ax.set_ylim(MATRIX_SIZE - 0.5, -0.5)
            
        else:
            # 正常模式：显示所有装饰元素
            # 设置子图标题
            network_title = NETWORK_TITLES[i]
            ax.set_title(network_title, fontsize=12, fontfamily='Times New Roman', 
                        fontweight='bold', pad=10)
            
            # 设置刻度标签
            tick_positions = range(MATRIX_SIZE)
            tick_labels = [str(i+1) for i in tick_positions]
            
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=8, fontfamily='Times New Roman')
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels, fontsize=8, fontfamily='Times New Roman')
            
            # 设置轴标签
            ax.set_xlabel('Node', fontsize=10.5, fontfamily='Times New Roman')
            ax.set_ylabel('Node', fontsize=10.5, fontfamily='Times New Roman')
            
            # 移除刻度线
            ax.tick_params(top=False, bottom=False, left=False, right=False)
    
    if not minimal_mode:
        # 添加总标题
        fig.suptitle(f"{group} - {band_name}", fontsize=14, fontfamily='Times New Roman', 
                    fontweight='bold', y=0.98)
        
        # 添加共享颜色条
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.93, 0.25, 0.015, 0.5])
        cbar = fig.colorbar(mappable, cax=cbar_ax)
        cbar.set_label('Connection Strength', fontsize=10.5, fontfamily='Times New Roman')
        cbar.ax.tick_params(labelsize=9)
        
        # 调整布局
        plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    else:
        # 最小化模式：调整布局以完全填充整个区域，无边距
        for ax in axes:
            ax.margins(0, 0)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.01)
    
    # 保存图片
    if save_fig:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        suffix = '_minimal' if minimal_mode else ''
        filename = f"{group}_{band_name}_all_networks{suffix}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0 if minimal_mode else 0.1)
        print(f"图片已保存至: {filepath}")
    
    print("绘图完成。")
    plt.show()

# --- 5. 示例用法 ---

if __name__ == "__main__":
    
    # 示例1: 绘制单个矩阵（正常模式）
    # 参数: 组别, 频段, 网络类型, 是否保存, 保存目录, 是否最小化模式
    # 示例1b: 绘制单个矩阵（最小化模式 - 仅保留主图）
    # plot_single_matrix('HC', 'Delta', 'lofc', save_fig=False, output_dir='single_matrices', 
    #                   minimal_mode=True)
    plot_single_matrix('AD', 'Delta', 'ahofc', save_fig=False, output_dir='single_matrices', 
                      minimal_mode=True)
    
    
    # 示例2: 绘制单个组别和频段的所有网络类型（正常模式）
    # plot_single_band_group_all_networks('HC', 'Delta', save_fig=False, 
    #                                    output_dir='band_group_networks', minimal_mode=False)
    
    # 示例2b: 绘制单个组别和频段的所有网络类型（最小化模式）
    # plot_single_band_group_all_networks('HC', 'Delta', save_fig=False, 
    #                                    output_dir='band_group_networks', minimal_mode=True)
    
    # 示例3: 批量绘制某个组别的所有频段（最小化模式）
    # for band in ['Delta', 'Theta', 'Alpha', 'Beta']:
    #     plot_single_band_group_all_networks('HC', band, save_fig=True, 
    #                                        output_dir='HC_all_bands', minimal_mode=True)
    
    # 示例4: 批量绘制所有组合（最小化模式）
    # for group in GROUPS:
    #     for band in BANDS_INFO.keys():
    #         for network in NETWORKS:
    #             plot_single_matrix(group, band, network, save_fig=True, 
    #                              output_dir='all_matrices', minimal_mode=True)
