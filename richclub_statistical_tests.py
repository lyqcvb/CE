import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, ttest_ind, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# 定义FC类型
fc_types = ['LOFC', 'HOFC', 'AHOFC']

# 定义频段
bands = ['Delta', 'Theta', 'Alpha', 'Beta']

# 定义组别
groups = ['HC', 'MCI', 'AD']

# 存储结果
results = []

print("=" * 100)
print("Rich-club 系数统计检验分析报告")
print("=" * 100)
print()

# 对每种FC类型进行分析
for fc_type in fc_types:
    print(f"\n{'='*100}")
    print(f"FC类型: {fc_type}")
    print(f"{'='*100}\n")
    
    # 读取数据
    input_file = f'richclub_individual_means_{fc_type}.csv'
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"警告: 文件 {input_file} 不存在，跳过")
        continue
    
    # 对每个频段进行分析
    for band in bands:
        print(f"\n--- 频段: {band} ---")
        
        # 存储各组数据和正态性结果
        group_data = {}
        normality_results = {}
        
        # 对每组进行正态性检验
        for group in groups:
            col_name = f'{band}_{group}'
            
            if col_name not in df.columns:
                print(f"  警告: 列 {col_name} 不存在")
                continue
            
            # 获取数据并去除 NaN
            group_values = df[col_name].dropna()
            group_data[group] = group_values
            
            # Shapiro-Wilk 正态性检验
            if len(group_values) >= 3:  # Shapiro-Wilk 至少需要 3 个样本
                stat, p_value = shapiro(group_values)
                normality_results[group] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'is_normal': p_value > 0.05
                }
                
                # 打印正态性检验结果
                normal_status = "✓ 满足正态性" if p_value > 0.05 else "✗ 不满足正态性"
                print(f"  FC={fc_type}, Band={band}, Group={group}")
                print(f"    Shapiro-Wilk 检验: W={stat:.4f}, p={p_value:.4f} -> {normal_status}")
            else:
                normality_results[group] = {
                    'statistic': np.nan,
                    'p_value': np.nan,
                    'is_normal': False
                }
                print(f"  FC={fc_type}, Band={band}, Group={group}")
                print(f"    ⚠ 样本量不足 (n={len(group_values)})，假设不满足正态性")
        
        # 进行两两比较
        group_list = [g for g in groups if g in group_data]
        for i in range(len(group_list)):
            for j in range(i + 1, len(group_list)):
                group1 = group_list[i]
                group2 = group_list[j]
                
                data1 = group_data[group1]
                data2 = group_data[group2]
                
                # 检查样本量
                if len(data1) < 2 or len(data2) < 2:
                    print(f"\n  ⚠ {group1} vs {group2}: 样本量不足，跳过")
                    continue
                
                # 判断是否都满足正态性
                both_normal = (normality_results[group1]['is_normal'] and 
                              normality_results[group2]['is_normal'])
                
                # 根据正态性选择检验方法
                if both_normal:
                    # 独立样本 t 检验
                    test_type = "Independent t-test"
                    stat, p_value = ttest_ind(data1, data2, equal_var=True)
                    
                    print(f"\n  {group1} vs {group2}:")
                    print(f"    两组均满足正态性 -> 使用 {test_type}")
                    print(f"    t-statistic = {stat:.4f}, p-value = {p_value:.4f}")
                    
                    if p_value < 0.001:
                        sig_label = "*** (p < 0.001)"
                    elif p_value < 0.01:
                        sig_label = "** (p < 0.01)"
                    elif p_value < 0.05:
                        sig_label = "* (p < 0.05)"
                    else:
                        sig_label = "n.s."
                    print(f"    显著性: {sig_label}")
                else:
                    # Mann-Whitney U 检验
                    test_type = "Mann-Whitney U test"
                    stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
                    
                    print(f"\n  {group1} vs {group2}:")
                    print(f"    至少一组不满足正态性 -> 使用 {test_type}")
                    print(f"    U-statistic = {stat:.4f}, p-value = {p_value:.4f}")
                    
                    if p_value < 0.001:
                        sig_label = "*** (p < 0.001)"
                    elif p_value < 0.01:
                        sig_label = "** (p < 0.01)"
                    elif p_value < 0.05:
                        sig_label = "* (p < 0.05)"
                    else:
                        sig_label = "n.s."
                    print(f"    显著性: {sig_label}")
                
                # 计算描述性统计
                mean1, std1 = data1.mean(), data1.std()
                mean2, std2 = data2.mean(), data2.std()
                print(f"    {group1}: Mean={mean1:.4f}, SD={std1:.4f}, n={len(data1)}")
                print(f"    {group2}: Mean={mean2:.4f}, SD={std2:.4f}, n={len(data2)}")
                
                # 保存结果
                results.append({
                    'FC_Type': fc_type,
                    'Band': band,
                    'Group1': group1,
                    'Group2': group2,
                    'Group1_n': len(data1),
                    'Group2_n': len(data2),
                    'Group1_Mean': mean1,
                    'Group1_SD': std1,
                    'Group2_Mean': mean2,
                    'Group2_SD': std2,
                    'Group1_Normal_p': normality_results[group1]['p_value'],
                    'Group2_Normal_p': normality_results[group2]['p_value'],
                    'Test_Type': test_type,
                    'Test_Statistic': stat,
                    'p_value': p_value,
                    'Significant': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'n.s.'
                })

# 创建结果 DataFrame
results_df = pd.DataFrame(results)

# 保存结果
output_file = 'richclub_显著性分析.csv'
results_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n\n{'='*100}")
print("分析完成！")
print(f"{'='*100}")
print(f"\n结果已保存至: {output_file}")
print(f"总共进行了 {len(results_df)} 次统计检验")
print(f"\n显著性统计:")
print(f"  p < 0.001 (***): {len(results_df[results_df['Significant'] == '***'])} 次")
print(f"  p < 0.01  (**):  {len(results_df[results_df['Significant'] == '**'])} 次")
print(f"  p < 0.05  (*):   {len(results_df[results_df['Significant'] == '*'])} 次")
print(f"  n.s.:            {len(results_df[results_df['Significant'] == 'n.s.'])} 次")

# 显示前几行结果
print(f"\n结果预览（前10行）:")
print(results_df.head(10).to_string())

# 显示显著性结果
significant_results = results_df[results_df['p_value'] < 0.05].sort_values('p_value')
if len(significant_results) > 0:
    print(f"\n\n{'='*100}")
    print(f"显著性结果汇总 (p < 0.05, 共 {len(significant_results)} 项)")
    print(f"{'='*100}")
    for idx, row in significant_results.head(20).iterrows():
        print(f"\n{row['FC_Type']} | {row['Band']} | {row['Group1']} vs {row['Group2']}")
        print(f"  {row['Test_Type']}: p = {row['p_value']:.4f} {row['Significant']}")
        print(f"  {row['Group1']}: {row['Group1_Mean']:.4f} ± {row['Group1_SD']:.4f}")
        print(f"  {row['Group2']}: {row['Group2_Mean']:.4f} ± {row['Group2_SD']:.4f}")
else:
    print(f"\n\n{'='*100}")
    print("未发现显著性差异 (p < 0.05)")
    print(f"{'='*100}")
