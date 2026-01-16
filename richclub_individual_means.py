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
    
    # 存储所有个体的数据
    all_subjects_data = []
    
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
            
            # 读取每个被试的数据并计算均值
            for csv_file in csv_files:
                try:
                    # 获取被试ID（文件名）
                    subject_id = os.path.basename(csv_file).replace('.csv', '')
                    
                    # 读取CSV文件（假设只有一行数据，用逗号分隔）
                    data = pd.read_csv(csv_file, header=None)
                    
                    # 如果是二维数据，取第一行
                    if len(data.shape) == 2:
                        values = data.iloc[0].values
                    else:
                        values = data.values
                    
                    # 计算该被试所有节点的 Rich-club 系数均值（忽略NaN）
                    mean_value = np.nanmean(values)
                    
                    # 存储数据
                    all_subjects_data.append({
                        'SubjectID': subject_id,
                        'Group': group,
                        'Band': band_name,
                        'RichClub_Mean': mean_value
                    })
                    
                except Exception as e:
                    print(f"读取文件 {csv_file} 时出错: {e}")
                    continue
            
            print(f"  {group} - {band_name}: 处理了 {len(csv_files)} 个被试")
    
    # 创建 DataFrame
    if all_subjects_data:
        df = pd.DataFrame(all_subjects_data)
        
        # 找出每个组别的最大被试数
        max_subjects = {}
        for group in groups:
            group_data = df[df['Group'] == group]
            # 对每个频段计算该组别的被试数
            max_count = 0
            for band_name in ['Delta', 'Theta', 'Alpha', 'Beta']:
                subset = group_data[group_data['Band'] == band_name]
                if len(subset) > max_count:
                    max_count = len(subset)
            max_subjects[group] = max_count
        
        # 创建一个DataFrame，每个组别为一行，列为频段
        data_dict = {'Group': []}
        for band_name in ['Delta', 'Theta', 'Alpha', 'Beta']:
            data_dict[band_name] = []
        
        # 为每个组别创建一行，包含该组别在每个频段的所有被试数据（用逗号分隔）
        for group in groups:
            data_dict['Group'].append(group)
            for band_name in ['Delta', 'Theta', 'Alpha', 'Beta']:
                subset = df[(df['Band'] == band_name) & (df['Group'] == group)].copy()
                if not subset.empty:
                    # 按SubjectID排序并获取所有被试的均值
                    subset = subset.sort_values('SubjectID').reset_index(drop=True)
                    values = subset['RichClub_Mean'].tolist()
                    # 将所有被试的值存储为列表
                    data_dict[band_name].append(values)
                else:
                    data_dict[band_name].append([])
        
        # 创建DataFrame（每个组别一行）
        pivot_df = pd.DataFrame(data_dict)
        
        # 找出所有组别中最大的被试数，以便对齐输出
        max_rows = 0
        for group in groups:
            for band_name in ['Delta', 'Theta', 'Alpha', 'Beta']:
                subset = df[(df['Band'] == band_name) & (df['Group'] == group)]
                if len(subset) > max_rows:
                    max_rows = len(subset)
        
        # 创建多级列名：第一行为频段，第二行为组别
        # 列结构：Group列 + (Delta-HC, Delta-MCI, Delta-AD, Theta-HC, Theta-MCI, ...)
        multi_columns = [('', 'Group')]
        for band_name in ['Delta', 'Theta', 'Alpha', 'Beta']:
            for group in groups:
                multi_columns.append((band_name, group))
        
        # 创建数据行
        output_rows = []
        for i in range(max_rows):
            row_data = [f'Subject_{i+1}']  # 第一列为被试编号
            for band_name in ['Delta', 'Theta', 'Alpha', 'Beta']:
                for group in groups:
                    subset = df[(df['Band'] == band_name) & (df['Group'] == group)].copy()
                    if not subset.empty:
                        subset = subset.sort_values('SubjectID').reset_index(drop=True)
                        if i < len(subset):
                            row_data.append(subset.iloc[i]['RichClub_Mean'])
                        else:
                            row_data.append(np.nan)
                    else:
                        row_data.append(np.nan)
            output_rows.append(row_data)
        
        # 创建DataFrame，使用多级列名
        final_df = pd.DataFrame(output_rows, columns=pd.MultiIndex.from_tuples(multi_columns))
        
        # 保存结果（保留多级列名）
        fc_type = fc_folder.replace('rich_club_', '').upper()
        output_file = f'richclub_individual_means_{fc_type}.csv'
        final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\n结果已保存至: {output_file}")
        print(f"数据形状: {final_df.shape[0]} 行（被试） × {final_df.shape[1]} 列（Group + 组别×频段）")
        
        # 统计每个组别每个频段的被试数量
        print(f"\n各组别各频段的被试数量:")
        for group in groups:
            counts = []
            for band_name in ['Delta', 'Theta', 'Alpha', 'Beta']:
                subset = df[(df['Band'] == band_name) & (df['Group'] == group)]
                counts.append(f"{band_name}:{len(subset)}")
            print(f"  {group}: {', '.join(counts)}")
        
        print(f"\n数据预览（前10行）:")
        # 设置pandas显示选项
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(final_df.head(10).to_string(index=False))

print(f"\n{'='*80}")
print("所有 Rich-club 个体均值计算完成！")
print(f"{'='*80}")
