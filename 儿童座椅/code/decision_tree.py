import pandas as pd
import numpy as np
from scipy.stats import zscore
# 使用 Z-score 方法检测异常值
df=pd.read_csv("py/儿童座椅/csv&excel/01儿童座椅数据_Eng.csv")
continuous_columns = df.select_dtypes(include=[np.number]).columns
z_scores = np.abs(zscore(df[continuous_columns]))
threshold = 3
outliers_z = np.where(z_scores > threshold)
print("Outliers detected using Z-score method:")
print(outliers_z)
# 使用 IQR 方法检测异常值
Q1 = df[continuous_columns].quantile(0.25)
Q3 = df[continuous_columns].quantile(0.75)
IQR = Q3 - Q1

outliers_iqr = (df[continuous_columns] < (Q1 - 1.5 * IQR)) | (df[continuous_columns] > (Q3 + 1.5 * IQR))
print("Outliers detected using IQR method:")
print(outliers_iqr)
# 合并两种方法检测到的异常值索引
outliers = np.unique(np.concatenate([outliers_z[0], np.where(outliers_iqr.any(axis=1))[0]]))

# 删除异常值
df = df.drop(outliers)
df = df.reset_index(drop=True)

# 导出清理后的数据到 CSV 文件
df.to_csv('py/儿童座椅/csv&excel/cleaned_df.csv', index=False, encoding='utf-8-sig')
print("Cleaned df has been exported to 'cleaned_df.csv'.")

# 生成二分类新变量 High
df['High'] = np.where(df['Child_seat_sales'] >= 8, 1, 0)
# 输出结果
print(df)
df.to_csv('py/儿童座椅/csv&excel/分类.csv', index=False, encoding='utf-8-sig')
print(len(df))
try:
    df1 = df.drop('Child_seat_sales', axis=1)
except KeyError as e:
    print(f"KeyError: {e}")
# print(df)
print(df1.info())
from sklearn.preprocessing import OneHotEncoder

# 'Shelf_location'  'Yes_or_no_city' 'Whether_it_is_in_the_United_States'
Shelf_location_column = df1['Shelf_location']
Yes_or_no_city_column = df1['Yes_or_no_city']
Whether_it_is_in_the_United_States_column = df1['Whether_it_is_in_the_United_States']

# 使用 OneHotEncoder 进行独热编码
onehot_encoder = OneHotEncoder(sparse=False)

# 对 'Shelf_location' 列进行独热编码
Shelf_location_encoded = onehot_encoder.fit_transform(Shelf_location_column.values.reshape(-1, 1))
Shelf_location_encoded_df = pd.DataFrame(Shelf_location_encoded, columns=onehot_encoder.get_feature_names_out(['Shelf_location']))

# 对 'Yes_or_no_city' 列进行独热编码，使用不同的列名
Yes_or_no_city_encoded = onehot_encoder.fit_transform(Yes_or_no_city_column.values.reshape(-1, 1))
Yes_or_no_city_encoded_df = pd.DataFrame(Yes_or_no_city_encoded, columns=onehot_encoder.get_feature_names_out(['Yes_or_no_city']))

# 对 'Whether_it_is_in_the_United_States' 列进行独热编码，使用不同的列名
Whether_it_is_in_the_United_States_encoded = onehot_encoder.fit_transform(Whether_it_is_in_the_United_States_column.values.reshape(-1, 1))
Whether_it_is_in_the_United_States_encoded_df = pd.DataFrame(Whether_it_is_in_the_United_States_encoded, columns=onehot_encoder.get_feature_names_out(['Whether_it_is_in_the_United_States']))

# 将独热编码的结果添加到数据框中
df1 = pd.concat([df1, Shelf_location_encoded_df, Yes_or_no_city_encoded_df,
                   Whether_it_is_in_the_United_States_encoded_df], axis=1)
df1 = df1.drop(['Shelf_location', 'Yes_or_no_city','Whether_it_is_in_the_United_States'], axis=1)  


X= df1.drop("High", axis=1).values
df2=df1.drop("High", axis=1)
y = df1["High"].copy().values
df2.info()
#分离训练集、验证集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)
print(len(X_train))
print(len(X_test))

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut

import numpy as np
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.metrics import accuracy_score,classification_report

# 创建决策树分类器
clf = DecisionTreeClassifier(random_state=42)

# 训练模型
clf.fit(X_train, y_train)

# 预测测试集
y_pred = clf.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# 打印分类报告
print(classification_report(y_test, y_pred, target_names=['0', '1']))

import pydotplus
from IPython.display import Image

# 导出决策树结构到DOT文件
dot_df = export_graphviz(clf, out_file=None, 
                           feature_names=df2.columns,  
                           class_names=['0', '1'],  
                           filled=True)  


# 使用pydotplus将DOT数据转换为图形
graph = pydotplus.graph_from_dot_data(dot_df)

# 将图形保存为PDF文件
graph.write_pdf("py/儿童座椅/pdf/decision_tree1.pdf")

# 将图形保存为图片文件
graph.write_png("py/儿童座椅/img/decision_tree/decision_tree1.png")

# 使用IPython.display在Notebook中显示决策树图像
Image(graph.create_png())
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# 剪枝
# 获取成本复杂度修剪路径
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# 初始化一个列表用于存储每个alpha对应的决策树和验证准确率
clfs = []
accuracy_scores = []

# 对每个alpha值进行修剪并评估性能
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    clfs.append(clf)
    accuracy_scores.append(accuracy)

# 找到最佳的alpha值
best_index = accuracy_scores.index(max(accuracy_scores))
best_ccp_alpha = ccp_alphas[best_index]
best_clf = clfs[best_index]

# 输出最佳alpha值和对应的准确率
print(f"Best ccp_alpha: {best_ccp_alpha}")
print(f"Best accuracy: {accuracy_scores[best_index]}")

# 打印最佳模型的分类报告
y_pred = best_clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['1', '0']))


import pydotplus
from IPython.display import Image

# 导出决策树结构到DOT文件
dot_df = export_graphviz(best_clf, out_file=None, 
                           feature_names=df2.columns,  
                           class_names=['0', '1'],  
                           filled=True)  


# 使用pydotplus将DOT数据转换为图形
graph = pydotplus.graph_from_dot_data(dot_df)

# 将图形保存为PDF文件
graph.write_pdf("py/儿童座椅/pdf/decision_tree2.pdf")

# 将图形保存为图片文件
graph.write_png("py/儿童座椅/img/decision_tree/decision_tree2.png")

# 使用IPython.display在Notebook中显示决策树图像
Image(graph.create_png())


import matplotlib.pyplot as plt
# 获取特征重要性
feature_importances = best_clf.feature_importances_

# 将特征重要性与特征名对应起来
feature_names = list(df2.columns)
importance_dict = dict(zip(feature_names, feature_importances))

# 根据特征重要性排序，并选择前八个
sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:8]
print(sorted_importance)


# 可视化前八个特征重要性
plt.figure(figsize=(14, 8))
bars=plt.barh(range(len(sorted_importance)), [val for key, val in sorted_importance], align='center', color=[(98/255, 158/255, 150/255)])#生成ppt所需的水平条形图，绿色的
plt.yticks(range(len(sorted_importance)), [key for key, val in sorted_importance],fontsize=7)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Top 8 Feature Importance in Decision Tree Classifier',fontsize=20)
plt.gca().invert_yaxis()  # 反转y轴，让重要性高的特征显示在顶部
for bar in bars:
    plt.text(
        bar.get_width(),   # x 坐标
        bar.get_y() + bar.get_height() / 2,  # y 坐标
        f'{bar.get_width():.4f}',  # 显示的文本
        va='center',  # 垂直对齐方式
        ha='left' if bar.get_width() >= 0 else 'right'  # 水平对齐方式
    )
# 保存图像
plt.savefig('py/儿童座椅/img/decision_tree/top_8_feature_importance.png',dpi=600)

# 显示图像
plt.show()