import pandas as pd
import numpy as np
import os
import glob
from collections import Counter

# 定义路径
core_nodes_folders = ['core_nodes_lofc', 'core_nodes_hofc', 'core_nodes_ahofc']

# 定义频段映射
band_mapping = {
    '0.5-4(Delta)': 'Delta',
    '4-8(Theta)': 'Theta',
    '8-13(Alpha)': 'Alpha',
    '13-40(Beta)': 'Beta'
}

# 定义组别
groups = ['HC', 'MCI', 'AD']

# 设置阈值（节点在组内被试中出现的比例）
THRESHOLD = 0.30  # 30%的被试中该节点为核心节点
# 可选阈值: 0.20 (20%), 0.25 (25%), 0.30 (30%), 0.40 (40%), 0.50 (50%)

print("="*100)
print(f"代表性核心节点提取分析 (阈值: {THRESHOLD*100}%)")
print("="*100)

# 存储所有结果
all_results = []

# 处理每种 FC 类型
for fc_folder in core_nodes_folders:
    fc_type = fc_folder.replace('core_nodes_', '').upper()
    
    print(f"\n{'='*100}")
    print(f"FC类型: {fc_type}")
    print(f"{'='*100}")
    
    if not os.path.exists(fc_folder):
        print(f"警告: 文件夹 {fc_folder} 不存在，跳过")
        continue
    
    # 处理每个频段
    for band_folder, band_name in band_mapping.items():
        print(f"\n--- 频段: {band_name} ---")
        
        # 处理每个组别
        for group in groups:
            group_path = os.path.join(fc_folder, group, band_folder)
            
            if not os.path.exists(group_path):
                print(f"  警告: {group_path} 不存在，跳过")
                continue
            
            # 获取所有CSV文件
            csv_files = glob.glob(os.path.join(group_path, "*_core_nodes.csv"))
            
            if not csv_files:
                print(f"  警告: {group_path} 中没有核心节点文件")
                continue
            
            # 统计每个节点被标记为核心节点的次数
            node_core_counts = Counter()
            node_names = {}  # 存储节点索引到节点名称的映射
            total_subjects = len(csv_files)
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    
                    # 获取核心节点（IsCoreNode=1）
                    core_nodes = df[df['IsCoreNode'] == 1]
                    
                    for _, row in core_nodes.iterrows():
                        node_idx = row['NodeIndex']
                        node_name = row['ChannelName']
                        node_core_counts[node_idx] += 1
                        node_names[node_idx] = node_name
                    
                except Exception as e:
                    print(f"  读取文件 {csv_file} 时出错: {e}")
                    continue
            
            # 计算每个节点的出现比例
            node_proportions = {node: count/total_subjects 
                              for node, count in node_core_counts.items()}
            
            # 筛选超过阈值的节点
            representative_nodes = {node: prop for node, prop in node_proportions.items() 
                                   if prop >= THRESHOLD}
            
            # 按比例降序排序
            sorted_nodes = sorted(representative_nodes.items(), key=lambda x: x[1], reverse=True)
            
            print(f"  {group}: {total_subjects} 个被试")
            
            if sorted_nodes:
                print(f"    代表性核心节点 (比例 ≥ {THRESHOLD*100}%): {len(sorted_nodes)} 个")
                for node_idx, proportion in sorted_nodes:
                    node_name = node_names[node_idx]
                    count = node_core_counts[node_idx]
                    print(f"      节点 {node_idx:2d} ({node_name:3s}): {count:2d}/{total_subjects} = {proportion*100:5.1f}%")
                    
                    # 保存结果
                    all_results.append({
                        'FC_Type': fc_type,
                        'Band': band_name,
                        'Group': group,
                        'NodeIndex': node_idx,
                        'ChannelName': node_name,
                        'Count': count,
                        'TotalSubjects': total_subjects,
                        'Proportion': proportion,
                        'Percentage': f"{proportion*100:.1f}%"
                    })
            else:
                print(f"    代表性核心节点 (比例 ≥ {THRESHOLD*100}%): 0 个")

# 保存结果
if all_results:
    results_df = pd.DataFrame(all_results)
    
    # 保存详细结果
    output_file = f'representative_core_nodes_threshold_{int(THRESHOLD*100)}.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n{'='*100}")
    print("分析完成！")
    print(f"{'='*100}")
    print(f"\n结果已保存至: {output_file}")
    print(f"共识别出 {len(results_df)} 个代表性核心节点")
    
    # 统计每个FC类型和频段的代表性节点数量
    print(f"\n代表性核心节点统计:")
    for fc_type in ['LOFC', 'HOFC', 'AHOFC']:
        fc_results = results_df[results_df['FC_Type'] == fc_type]
        if not fc_results.empty:
            print(f"\n  {fc_type}:")
            for band in ['Delta', 'Theta', 'Alpha', 'Beta']:
                band_results = fc_results[fc_results['Band'] == band]
                if not band_results.empty:
                    print(f"    {band}:")
                    for group in groups:
                        group_results = band_results[band_results['Group'] == group]
                        print(f"      {group}: {len(group_results)} 个节点")
    
    # 创建汇总表格（按频段-组别-节点组织）
    print(f"\n创建汇总表格...")
    
    # 为每个FC类型创建一个汇总表
    for fc_type in ['LOFC', 'HOFC', 'AHOFC']:
        fc_results = results_df[results_df['FC_Type'] == fc_type]
        
        if fc_results.empty:
            continue
        
        # 创建透视表
        summary_rows = []
        
        for band in ['Delta', 'Theta', 'Alpha', 'Beta']:
            for group in groups:
                subset = fc_results[(fc_results['Band'] == band) & (fc_results['Group'] == group)]
                
                if not subset.empty:
                    # 按比例排序
                    subset_sorted = subset.sort_values('Proportion', ascending=False)
                    
                    # 获取节点列表和比例列表
                    nodes_list = ', '.join([f"{row['ChannelName']}({row['Percentage']})" 
                                          for _, row in subset_sorted.iterrows()])
                    
                    summary_rows.append({
                        'Band': band,
                        'Group': group,
                        'NumNodes': len(subset),
                        'RepresentativeNodes': nodes_list
                    })
                else:
                    summary_rows.append({
                        'Band': band,
                        'Group': group,
                        'NumNodes': 0,
                        'RepresentativeNodes': 'None'
                    })
        
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_file = f'representative_core_nodes_summary_{fc_type}_threshold_{int(THRESHOLD*100)}.csv'
            summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
            print(f"  {fc_type} 汇总表已保存至: {summary_file}")
    
    # 显示数据预览
    print(f"\n数据预览（前20行）:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(results_df.head(20).to_string(index=False))
    
else:
    print("\n未识别出任何代表性核心节点")

print(f"\n{'='*100}")
