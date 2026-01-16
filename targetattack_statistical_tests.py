import pandas as pd
import numpy as np
from scipy import stats
import os
import glob

# 定义路径
target_attack_folders = ['target_attack_lofc', 'target_attack_hofc', 'target_attack_ahofc']

# 定义频段映射
band_mapping = {
    '0.5-4(Delta)': 'Delta',
    '4-8(Theta)': 'Theta',
    '8-13(Alpha)': 'Alpha',
    '13-40(Beta)': 'Beta'
}

# 定义组别
groups = ['HC', 'MCI', 'AD']

# 定义组别比较
comparisons = [('HC', 'MCI'), ('HC', 'AD'), ('MCI', 'AD')]

# 存储所有结果
all_results = []

print("="*100)
print("Target Attack 显著性差异分析")
print("="*100)

# 处理每种 FC 类型
for fc_folder in target_attack_folders:
    fc_type = fc_folder.replace('target_attack_', '').upper()
    
    print(f"\n{'='*100}")
    print(f"FC类型: {fc_type}")
    print(f"{'='*100}")
    
    if not os.path.exists(fc_folder):
        print(f"警告: 文件夹 {fc_folder} 不存在，跳过")
        continue
    
    # 处理每个频段
    for band_folder, band_name in band_mapping.items():
        print(f"\n--- 频段: {band_name} ---")
        
        # 读取该频段下各组别的所有被试数据
        group_data = {}
        
        for group in groups:
            band_path = os.path.join(fc_folder, group, band_folder)
            
            if not os.path.exists(band_path):
                print(f"警告: {band_path} 不存在，跳过")
                continue
            
            # 获取该频段下的所有CSV文件
            csv_files = glob.glob(os.path.join(band_path, "*.csv"))
            
            if not csv_files:
                print(f"警告: {band_path} 中没有CSV文件")
                continue
            
            # 读取所有被试的数据
            subjects_data = []
            for csv_file in csv_files:
                try:
                    data = pd.read_csv(csv_file, header=None)
                    if len(data.shape) == 2:
                        values = data.iloc[0].values
                    else:
                        values = data.values
                    subjects_data.append(values)
                except Exception as e:
                    print(f"读取文件 {csv_file} 时出错: {e}")
                    continue
            
            if subjects_data:
                # 转换为numpy数组: (n_subjects, n_attack_steps)
                group_data[group] = np.array(subjects_data)
                print(f"  {group}: {len(subjects_data)} 个被试, {group_data[group].shape[1]} 个攻击步骤")
        
        # 如果所有组别都有数据，进行统计检验
        if len(group_data) == 3:
            # 获取攻击步骤数
            n_steps = group_data['HC'].shape[1]
            
            # 对每个攻击步骤进行组间比较
            for step in range(n_steps):
                step_num = step + 1  # 攻击步骤从1开始
                
                # 对每对组别进行比较
                for group1, group2 in comparisons:
                    # 提取该步骤的数据
                    data1 = group_data[group1][:, step]
                    data2 = group_data[group2][:, step]
                    
                    # 去除NaN值
                    data1_clean = data1[~np.isnan(data1)]
                    data2_clean = data2[~np.isnan(data2)]
                    
                    if len(data1_clean) < 3 or len(data2_clean) < 3:
                        continue
                    
                    # 计算描述统计
                    mean1 = np.mean(data1_clean)
                    std1 = np.std(data1_clean, ddof=1)
                    mean2 = np.mean(data2_clean)
                    std2 = np.std(data2_clean, ddof=1)
                    
                    # 正态性检验
                    _, p_normal1 = stats.shapiro(data1_clean)
                    _, p_normal2 = stats.shapiro(data2_clean)
                    
                    # 选择统计检验方法
                    if p_normal1 > 0.05 and p_normal2 > 0.05:
                        # 两组都满足正态性，使用独立样本t检验
                        stat, p_value = stats.ttest_ind(data1_clean, data2_clean)
                        test_type = "Independent t-test"
                        test_stat_name = "t-statistic"
                    else:
                        # 至少一组不满足正态性，使用Mann-Whitney U检验
                        stat, p_value = stats.mannwhitneyu(data1_clean, data2_clean, alternative='two-sided')
                        test_type = "Mann-Whitney U test"
                        test_stat_name = "U-statistic"
                    
                    # 判断显著性
                    if p_value < 0.001:
                        significance = "***"
                    elif p_value < 0.01:
                        significance = "**"
                    elif p_value < 0.05:
                        significance = "*"
                    else:
                        significance = "n.s."
                    
                    # 存储结果
                    result = {
                        'FC_Type': fc_type,
                        'Band': band_name,
                        'Attack_Step': step_num,
                        'Group1': group1,
                        'Group2': group2,
                        'Group1_n': len(data1_clean),
                        'Group2_n': len(data2_clean),
                        'Group1_Mean': mean1,
                        'Group1_SD': std1,
                        'Group2_Mean': mean2,
                        'Group2_SD': std2,
                        'Group1_Normal_p': p_normal1,
                        'Group2_Normal_p': p_normal2,
                        'Test_Type': test_type,
                        'Test_Statistic': stat,
                        'p_value': p_value,
                        'Significance': significance
                    }
                    
                    all_results.append(result)
                    
                    # 只打印显著的结果
                    if significance != "n.s.":
                        print(f"  步骤 {step_num}: {group1} vs {group2}")
                        print(f"    {test_type}: p = {p_value:.4f} {significance}")
                        print(f"    {group1}: {mean1:.4f} ± {std1:.4f}")
                        print(f"    {group2}: {mean2:.4f} ± {std2:.4f}")

# 创建结果DataFrame
if all_results:
    results_df = pd.DataFrame(all_results)
    
    # 保存结果
    output_file = 'targetattack_显著性分析.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n{'='*100}")
    print("分析完成！")
    print(f"{'='*100}")
    print(f"\n结果已保存至: {output_file}")
    print(f"总共进行了 {len(results_df)} 次统计检验")
    
    # 统计显著性结果
    sig_counts = results_df['Significance'].value_counts()
    print(f"\n显著性统计:")
    for sig_level in ['***', '**', '*', 'n.s.']:
        count = sig_counts.get(sig_level, 0)
        print(f"  {sig_level}: {count} 次")
    
    # 显示结果预览
    print(f"\n结果预览（前10行）:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(results_df.head(10).to_string(index=False))
    
    # 汇总各频段的显著性结果
    print(f"\n{'='*100}")
    print("显著性结果汇总 (p < 0.05)")
    print(f"{'='*100}")
    
    # 按FC类型和频段分组显示显著结果
    sig_results = results_df[results_df['Significance'] != 'n.s.'].copy()
    
    if not sig_results.empty:
        for fc_type in ['LOFC', 'HOFC', 'AHOFC']:
            fc_sig = sig_results[sig_results['FC_Type'] == fc_type]
            if not fc_sig.empty:
                print(f"\n{fc_type}:")
                for band in ['Delta', 'Theta', 'Alpha', 'Beta']:
                    band_sig = fc_sig[fc_sig['Band'] == band]
                    if not band_sig.empty:
                        print(f"  {band} 频段: {len(band_sig)} 个显著差异")
                        # 统计每对组别的显著差异数
                        for group1, group2 in comparisons:
                            pair_sig = band_sig[(band_sig['Group1'] == group1) & (band_sig['Group2'] == group2)]
                            if not pair_sig.empty:
                                steps = pair_sig['Attack_Step'].tolist()
                                print(f"    {group1} vs {group2}: 步骤 {steps}")
    else:
        print("\n未发现显著差异 (p < 0.05)")

else:
    print("\n未生成任何结果")

print(f"\n{'='*100}")
