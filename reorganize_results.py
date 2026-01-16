import pandas as pd
import numpy as np
from scipy import stats

# 读取原始数据
df = pd.read_csv('all_scalar_metrics_summary.csv')

# 定义指标类型和FC矩阵类型
metric_types = ['MeanConn', 'Eglob', 'Cluster']
fc_types = ['LOFC', 'HOFC', 'aHOFC']

# 定义频段顺序
bands = ['Delta', 'Theta', 'Alpha', 'Beta']

# 定义组别
groups = ['HC', 'MCI', 'AD']

# 创建结果字典
results = {'Band': bands}

# 按指标类型 -> FC类型 -> 组别的顺序组织列
# 这样相同指标的不同FC矩阵会相邻排列
for metric_type in metric_types:
    for fc_type in fc_types:
        metric = f"{metric_type}_{fc_type}"
        
        for group in groups:
            mean_col_name = f"{metric}_{group}_Mean"
            sem_col_name = f"{metric}_{group}_SEM"
            mean_values = []
            sem_values = []
            
            for band in bands:
                # 获取特定频段、特定组别、特定指标的数据
                band_group_data = df[(df['Band'] == band) & (df['Group'] == group)][metric]
                
                # 计算均值
                mean_value = band_group_data.mean()
                mean_values.append(mean_value)
                
                # 计算标准误差 (SEM = Standard Error of Mean)
                sem_value = band_group_data.sem()
                sem_values.append(sem_value)
            
            results[mean_col_name] = mean_values
            results[sem_col_name] = sem_values

# 创建 DataFrame
results_df = pd.DataFrame(results)

# 保存结果
output_file = '重组数据_按指标分组.csv'
results_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print("数据重组完成！")
print(f"结果已保存至: {output_file}")
print(f"\n数据预览（前几列）:")
# 只显示前10列以便阅读
print(results_df.iloc[:, :10].to_string())

# 显示数据形状
print(f"\n数据形状: {results_df.shape[0]} 行 × {results_df.shape[1]} 列")
print(f"频段: {', '.join(bands)}")
print(f"指标类型: {', '.join(metric_types)}")
print(f"FC矩阵类型: {', '.join(fc_types)}")
print(f"每个指标包含: 均值(Mean) + 标准误差(SEM)")
print(f"每个指标的组别: {', '.join(groups)}")
print(f"列排列顺序: 指标类型 -> FC类型 -> 组别")
print(f"每个完整指标共有: {len(groups)} × 2 = {len(groups) * 2} 列（均值+SEM）")
