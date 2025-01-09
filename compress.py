import pandas as pd

# 定义文件路径
# 定义文件路径
file1 = './visuals/task003_20241217_183608gt.xlsx'
file2 = './visuals/task003_20241217_183742gt.xlsx'
file3 = './visuals/task003_20241217_183929gt.xlsx'

# 读取每个文件到一个 DataFrame
df1 = pd.read_excel(file1)
df2 = pd.read_excel(file2)
df3 = pd.read_excel(file3)

# 检查每个 DataFrame 是否包含 'gt' 列
for i, df in enumerate([df1, df2, df3], start=1):
    if 'GroundTruth' not in df.columns:
        raise ValueError(f"文件 {i} 中缺少 'gt' 列。")

# 验证所有 'gt' 列是否相同
if not (df1['GroundTruth'].equals(df2['GroundTruth']) and df1['GroundTruth'].equals(df3['GroundTruth'])):
    raise ValueError("所有文件中的 'gt' 列不完全相同。请确保它们一致后再进行合并。")

# 保留第一个 DataFrame 的 'gt' 列，删除其他 DataFrame 中的 'gt' 列
df2 = df2.drop(columns=['GroundTruth'])
df3 = df3.drop(columns=['GroundTruth'])

# 合并所有 DataFrame
merged_df = pd.concat([df1, df2, df3], axis=1)

# 检查合并后的 DataFrame 是否有重复的列名（除了 'gt'）
# 如果有重复，可以选择重命名或处理
if merged_df.columns.duplicated().any():
    raise ValueError("合并后存在重复的列名。请检查并重命名重复的列。")

# 导出合并后的 DataFrame 到新的 Excel 文件
output_file = 'test_final_features.xlsx'
merged_df.to_excel(output_file, index=False)

print(f"数据已成功合并并保存到 {output_file}")
