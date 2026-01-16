import pandas as pd
import numpy as np
import os
import glob

# 定义路径
rich_club_folders = ['rich_club_lofc', 'rich_club_hofc', 'rich_club_ahofc']

# 定义频段映射
band_mapping = {
    '0.5-4(Delta)': 'Delta',
    '4-8(Theta)': 'Theta',
    '8-13(Alpha)': 'Alpha',
    '13-40(Beta)': 'Beta'
}

# 定义组别
groups = ['HC', 'MCI', 'AD']

# 为每种 FC 类型创建一个汇总结果
for fc_folder in rich_club_folders:
    print(f"\n{'='*80}")
    print(f"处理 {fc_folder}")
    print(f"{'='*80}")
    
    if not os.path.exists(fc_folder):
        print(f"警告: 文件夹 {fc_folder} 不存在，跳过")
        continue
    
    # 存储所有数据
    all_data = {}
    
    # 遍历每个组别
    for group in groups:
        group_path = os.path.join(fc_folder, group)
        
        if not os.path.exists(group_path):
            print(f"警告: {group_path} 不存在，跳过")
            continue
        
        # 遍历每个频段
        for band_folder, band_name in band_mapping.items():
            band_path = os.path.join(group_path, band_folder)
            
            if not os.path.exists(band_path):
                print(f"警告: {band_path} 不存在，跳过")
                continue
            
            # 获取该频段下的所有CSV文件
            csv_files = glob.glob(os.path.join(band_path, "*.csv"))
            
            if not csv_files:
                print(f"警告: {band_path} 中没有CSV文件")
                continue
            
            # 读取所有CSV文件并计算平均值
            all_subjects_data = []
            
            for csv_file in csv_files:
                try:
                    # 读取CSV文件（假设只有一行数据，用逗号分隔）
                    data = pd.read_csv(csv_file, header=None)
                    
                    # 如果是二维数据，取第一行
                    if len(data.shape) == 2:
                        values = data.iloc[0].values
                    else:
                        values = data.values
                    
                    all_subjects_data.append(values)
                except Exception as e:
                    print(f"读取文件 {csv_file} 时出错: {e}")
                    continue
            
            if all_subjects_data:
                # 计算该组别、该频段的平均 rich-club 系数
                mean_richclub = np.mean(all_subjects_data, axis=0)
                
                # 存储数据
                key = f"{band_name}_{group}"
                all_data[key] = mean_richclub
                
                print(f"  {group} - {band_name}: {len(all_subjects_data)} 个被试, 节点数={len(mean_richclub)}")
    
    # 生成汇总表格
    if all_data:
        # 找出最大节点数
        max_nodes = max(len(v) for v in all_data.values())
        
        # 创建节点列（度数 k）
        k_values = list(range(1, max_nodes + 1))
        
        # 创建DataFrame
        result_dict = {'k (Node Degree)': k_values}
        
        # 按频段和组别顺序添加列
        for band_name in ['Delta', 'Theta', 'Alpha', 'Beta']:
            for group in groups:
                key = f"{band_name}_{group}"
                if key in all_data:
                    data = all_data[key]
                    # 如果数据长度不足，用 NaN 填充
                    padded_data = list(data) + [np.nan] * (max_nodes - len(data))
                    result_dict[key] = padded_data
        
        # 创建DataFrame
        result_df = pd.DataFrame(result_dict)
        
        # 保存结果
        fc_type = fc_folder.replace('rich_club_', '').upper()
        output_file = f'richclub_summary_{fc_type}.csv'
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\n结果已保存至: {output_file}")
        print(f"数据形状: {result_df.shape[0]} 行 × {result_df.shape[1]} 列")
        print(f"\n数据预览:")
        print(result_df.head(20).to_string())

print(f"\n{'='*80}")
print("所有 Rich-club 数据处理完成！")
print(f"{'='*80}")
