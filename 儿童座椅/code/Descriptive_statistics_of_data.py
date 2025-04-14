import pandas as pd
import numpy as np

# df=pd.read_csv("py/儿童座椅/csv&excel/01儿童座椅数据_Eng.csv",encoding="GB2312")
# df.to_csv('py/儿童座椅/csv&excel/01儿童座椅数据_Eng.csv', index=True, encoding='utf-8-sig')
df=pd.read_csv("py/儿童座椅/csv&excel/01儿童座椅数据_Eng.csv")
#快速查看数据结构

# print(df.head())
# print(df.info())
# print(df.describe())
# 计算连续型变量的统计值
continuous_columns = df.select_dtypes(include=[np.number]).columns
continuous_stats = df[continuous_columns].describe().transpose()
continuous_stats['median'] = df[continuous_columns].median()
continuous_stats['1st_quartile'] = df[continuous_columns].quantile(0.25)
continuous_stats['3rd_quartile'] = df[continuous_columns].quantile(0.75)
print("Continuous variables statistics:")
print(continuous_stats)
continuous_stats.to_csv('py/儿童座椅/csv&excel/continuous_stats.csv', index=True, encoding='utf-8-sig')
import matplotlib.pyplot as plt

# 选择离散型变量
categorical_columns = df.select_dtypes(include=['object']).columns

# 计算每个离散型变量的分类频数表
frequency_tables = {}
for column in categorical_columns:
    frequency_tables[column] = df[column].value_counts()

# 将分类频数表转换为DataFrame
frequency_df = pd.DataFrame(frequency_tables)

# 将索引重置为分类名称
frequency_df = frequency_df.reset_index().rename(columns={'index': 'Category'})

# 输出结果
print(frequency_df)

# 导出分类频数表为CSV文件
frequency_df.to_csv('py/儿童座椅/csv&excel/categorical_frequency.csv', index=False, encoding='utf-8-sig')


# 定义RGB颜色
colors = [(98,158,150), (240,229,207), (176,206,202)]  # 绿、黄、浅黄

# 将RGB值归一化到[0, 1]
colors = [(r / 255, g / 255, b / 255) for (r, g, b) in colors]

# 绘制每个离散型变量的饼图
for column in categorical_columns:
    # 计算分类频数
    freq = df[column].value_counts()
    
    # 绘制饼图
    plt.figure(figsize=(8, 8))
    plt.pie(freq, labels=[f'{label} ({count})' for label, count in zip(freq.index, freq)], autopct='%1.1f%%', startangle=140, colors=colors[:len(freq)])
    plt.title(f'Distribution of {column}')
    
    # 保存图像
    plt.savefig(f'py/儿童座椅/img/Descriptive_statistics_of_data/{column}_distribution_pie_chart.png', dpi=600)
    
    # 显示图像
    plt.show()