import pymysql
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# 连接到MySQL数据库
conn = pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    password='123',
    database='scene',
    charset='utf8mb4'
)

# 读取评分数据
image_scores_beautiful_df = pd.read_sql(
    "SELECT image_id, score FROM image_scores_beautiful_s", 
    con=conn
)

# 将评分数据存储到字典中
scores = dict(zip(image_scores_beautiful_df['image_id'], 
                 image_scores_beautiful_df['score']))

# 读取比赛数据并过滤平局样本
data = pd.read_sql("""
    SELECT left_id, right_id, winner_label_encoded 
    FROM pp2_combined 
    WHERE category = 'beautiful' 
      AND winner_label_encoded != 0  -- 排除平局样本
    """, con=conn)

# 初始化统计变量
correct_predictions = 0
total_predictions = 0
misjudgments = defaultdict(int)
# 在初始化部分添加
misjudgment_diffs = []  # 存储误判对决的分数差值

for _, row in data.iterrows():
    left_id = row['left_id']
    right_id = row['right_id']
    result = row['winner_label_encoded']  # 1=左胜，2=右胜
    
    # 获取评分（缺失值处理为0）
    left_score = scores.get(left_id, 0)
    right_score = scores.get(right_id, 0)
    
    # 预测结果（只考虑胜负）
    predicted_result = 1 if left_score > right_score else 2

    # 统计结果
    if predicted_result == result:
        correct_predictions += 1
    else:
        # # 记录误判类型
        # if result == 1 and predicted_result == 2:
        #     misjudgments["实际左胜预测为右胜"] += 1
        # elif result == 2 and predicted_result == 1:
        #     misjudgments["实际右胜预测为左胜"] += 1
         # 记录误判类型
        if result == 1 and predicted_result == 2:
            misjudgments["实际左胜预测为右胜"] += 1
            # 计算真实左分与右分的差值（左胜本应左分高）
            diff = left_score - right_score
        elif result == 2 and predicted_result == 1:
            misjudgments["实际右胜预测为左胜"] += 1
            # 计算真实右分与左分的差值（右胜本应右分高）
            diff = right_score - left_score
    
        misjudgment_diffs.append(diff)  # 存储实际胜方与对方的分数差
    total_predictions += 1

# 计算并输出结果
if total_predictions > 0:
    accuracy = correct_predictions / total_predictions * 100
else:
    accuracy = 0.0

print(f"有效对决数量（排除平局）: {total_predictions}")
print(f"预测准确率: {accuracy:.2f}%")
print("\n误判类型分布:")
for desc, count in misjudgments.items():
    print(f"{desc}: {count}次")

# 在结果输出部分添加统计
print("\n误判对决分差统计（实际胜方分数 - 败方分数）：")
if misjudgment_diffs:
    df = pd.DataFrame(misjudgment_diffs, columns=['分差'])
    
    print(f"平均分差: {df['分差'].mean():.2f}")
    print(f"中位数分差: {df['分差'].median():.2f}")
    print(f"最大正向分差: {df['分差'].max():.2f}（实际胜方分数更高）")
    print(f"最大负向分差: {df['分差'].min():.2f}（实际败方分数更高）")
    print("\n分差分布直方图（单位：分）：")
    print(df['分差'].describe(percentiles=[0.25, 0.5, 0.75, 0.9]))
    
    # 绘制分差分布可视化
    plt.figure(figsize=(10,4))
    sns.histplot(data=df, x='分差', bins=30, kde=True)
    plt.title('误判对决实际胜负分数差值分布')
    plt.xlabel('实际胜方分数 - 败方分数')
    plt.ylabel('出现次数')
    plt.show()
else:
    print("无误判案例")
conn.close()