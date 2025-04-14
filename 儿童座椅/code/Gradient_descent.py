import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

class LinearRegressionGD(BaseEstimator, RegressorMixin):
    def __init__(self, learning_rate=1e-7, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        self.theta_ = np.random.randn(X.shape[1], 1)
        m = X.shape[0]
        for iteration in range(self.n_iterations):
            gradients = 2/m * X.T.dot(X.dot(self.theta_) - y)
            self.theta_ -= self.learning_rate * gradients
        return self

    def predict(self, X):
        return X.dot(self.theta_)
    

    def predict(self, X):
        return X.dot(self.theta_)
def load_data(file_path):
    df = pd.read_csv(file_path)
    print(len(df))
    try:
        df = df.drop('High', axis=1)
    except KeyError as e:
        print(f"KeyError: {e}")
    return df

def data_preprocess(df):
    # 提取特征和目标变量
    X = df.iloc[:, 1:].values  # 第一列之后的列作为特征
    y = df.iloc[:, 0].values   # 第一列作为目标变量

    # 对分类特征进行独热编码（One-Hot Encoding）
    categorical_cols = [5, 8, 9]  # 分类特征的列索引

    encoder = OneHotEncoder(sparse=False)
    X_categorical = encoder.fit_transform(X[:, categorical_cols])

    # 获取独热编码后的分类特征名称
    categorical_features = []
    for i, category in enumerate(categorical_cols):
        categories = encoder.categories_[i]
        categorical_features.extend([f"{df.columns[category+1]}_{cat}" for cat in categories])
    print(categorical_features)
    
    del categorical_features[2]
    del categorical_features[2]
    del categorical_features[3]
    # 获取最终的特征名称，包括原始特征和独热编码后的特征
    final_feature_names = list(df.columns[1:6]) + list(df.columns[7:9]) + categorical_features

    # 去除原始的分类特征列
    X = np.delete(X, categorical_cols, axis=1)

    # 对特征进行标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 并将独热编码后的特征与原始特征合并
    X = np.hstack((X, X_categorical))

    # 特征筛选
    X = np.delete(X, [9,10,12], axis=1)

    # 分割数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 添加偏置项 x0 = 1 到训练集和测试集中的特征矩阵中
    X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    return X_train_b, X_test_b, y_train, y_test, final_feature_names

def train_model(X_train_b, y_train, learning_rate, n_iterations):
    # 初始权重
    theta = np.random.randn(X_train_b.shape[1], 1)

    # 梯度下降算法
    for iteration in range(n_iterations):
        gradients = 2/X_train_b.shape[0] * X_train_b.T.dot(X_train_b.dot(theta) - y_train.reshape(-1, 1))
        theta = theta - learning_rate * gradients
        
    # 在测试集上进行预测
    y_pred = X_test_b.dot(theta)
    
    # 计算均方误差
    mse = mean_squared_error(y_test, y_pred)
    
    return theta, mse
def grid_search_model(X_train_b, y_train, X_test_b, y_test, param_grid):
    # 使用网格搜索寻找最佳参数
    grid_search = GridSearchCV(LinearRegressionGD(), param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_b, y_train)

    best_params = grid_search.best_params_
    # 使用最佳参数预测
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_b)

    # 计算均方误差
    mse = mean_squared_error(y_test, y_pred)
    return best_model, mse, best_params

def plot_n_iterations(X_train_b, y_train, X_test_b, y_test, n_iterations_list, learning_rate,title):
    # 初始化模型
    model = LinearRegressionGD(learning_rate = learning_rate)

    # 用于存储每个迭代次数下的训练集和测试集的均方误差
    train_errors = []
    test_errors = []

    for n_iterations in n_iterations_list:
        model.n_iterations = n_iterations  # 更新迭代次数
        model.fit(X_train_b, y_train)  # 训练模型
        
        # 计算在训练集上的均方误差
        y_train_pred = model.predict(X_train_b)
        train_error = mean_squared_error(y_train, y_train_pred)
        train_errors.append(train_error)
        
        # 计算在测试集上的均方误差
        y_test_pred = model.predict(X_test_b)
        test_error = mean_squared_error(y_test, y_test_pred)
        test_errors.append(test_error)
    # 绘制学习曲线
    plt.figure(figsize=(10, 6))
    plt.plot(n_iterations_list, train_errors, label='Train MSE')
    plt.plot(n_iterations_list, test_errors, label='Test MSE')
    plt.title(title)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.savefig("py/儿童座椅/img/Gradient_descent/Learning_Curve_for_Learning_Rate=0.01.png",dpi=600)
    plt.show()



def plot_learning_curve(estimator, title, X, y, param_name, param_range, cv=None, n_jobs=None, 
                        train_sizes=np.linspace(0.1, 1.0, 5)):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training Examples")
    plt.ylabel("Mean Squared Error")

    for param_value in param_range:
        # 动态设置参数值
        current_estimator = estimator.set_params(**{param_name: param_value})
        
        # 计算学习曲线数据
        train_sizes, train_scores, test_scores = learning_curve(
            current_estimator, X, y, cv=cv, n_jobs=n_jobs, 
            train_sizes=train_sizes, scoring='neg_mean_squared_error'
        )
        
        # 计算均值并取反（因为scoring是负误差）
        train_scores_mean = -np.mean(train_scores, axis=1)
        test_scores_mean = -np.mean(test_scores, axis=1)
        
        # 绘制训练集和测试集曲线
        plt.plot(train_sizes, train_scores_mean, '--', label=f"{param_name}={param_value} (Train)")
        plt.plot(train_sizes, test_scores_mean, '-o', label=f"{param_name}={param_value} (Test)")

    # 图例和布局
    plt.legend(loc="best", prop={'size': 7})
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"py/儿童座椅/img/Gradient_descent/{title}.png",dpi=600)
    plt.show()
    
def plot_feature_influence(final_feature_names, best_model):
    # 提取特征名称
    feature_names = final_feature_names

    # 提取模型训练后的参数值
    coefficients = best_model.theta_[1:].flatten()
    print(best_model.theta_)
    print(coefficients)
    # 绘制特征对应的参数条形图
    plt.figure(figsize=(14, 6))
    bars = plt.barh(feature_names, coefficients, color=['b' if w >= 0 else 'r' for w in coefficients])
    plt.xlabel('Parameter Value')
    plt.ylabel('Features')
    plt.title('Impact of Features on Dependent Variable')
    plt.grid(axis='x')
    # 在条形图上显示数值
    for bar in bars:
        plt.text(
            bar.get_width(),   # x 坐标
            bar.get_y() + bar.get_height() / 2,  # y 坐标
            f'{bar.get_width():.2f}',  # 显示的文本
            va='center',  # 垂直对齐方式
            ha='left' if bar.get_width() >= 0 else 'right'  # 水平对齐方式
        )
    plt.yticks(fontsize=7)
    plt.savefig('py/儿童座椅/img/Gradient_descent/Impact of Features on Dependent Variable2.png',dpi=600)
    plt.show()
    
if __name__ == '__main__':
    # 加载数据
    df = load_data('py/儿童座椅/csv&excel/分类.csv')
    # 数据预处理
    X_train_b, X_test_b, y_train, y_test, final_feature_names = data_preprocess(df)
    # 设定学习率和迭代次数
    learning_rate = 0.00001
    n_iterations = 100000
    # 训练模型(无网格搜索)
    theta1, mse1 = train_model(X_train_b, y_train, learning_rate, n_iterations)
    # 超参数show
    print(f'学习率: {learning_rate}, 迭代次数: {n_iterations}')
    # 打印测试集的均方误差
    print("测试集均方误差(MSE):", mse1)
    # # 打印训练后的参数
    # print("训练后的参数(theta):", theta1)
    
    # -------------------------------------------
     
    # 设定网格搜索的参数范围
    param_grid = {
        'learning_rate': [1e-7, 1e-6, 1e-5,1e-4, 1e-3, 1e-2,1e-1],
        # 'n_iterations': [100,1000, 10000, 100000,1000000]
        # 'n_iterations': [2000, 3000,4000, 5000, 6000,7000,8000, 9000,10000]
        'n_iterations': [1500,2000,2500, 3000,3500,4000,4500,5000]
    }
    # 将一维转二维
    y_train = y_train.reshape(-1, 1)
    # 使用网格搜索寻找最佳参数
    best_model, mse2, best_param = grid_search_model(X_train_b, y_train, X_test_b, y_test, param_grid)
    # 打印最佳参数和mse
    print("最佳参数:", best_param)
    print("测试集均方误差(MSE):", mse2)
    # # 打印训练后的参数
    # print("训练后的参数(theta):", best_model.theta_)
    
    # -------------------------------------------
    
    # 定义迭代次数范围
    n_iterations_list = [500,1000,1500,2000,2500, 3000,3500,4000,4500,5000]
    learning_rate = 0.01  # 学习率
    title = 'Learning Curve for Learning Rate=0.01'
    plot_n_iterations(X_train_b, y_train, X_test_b, y_test, n_iterations_list, learning_rate, title)
    
    # -------------------------------------------
    # Plot learning curves
    param_grid_lr = [  1e-3, 1e-2,1e-1]
    param_grid_iter = [3000]

    title = "Learning Curve for Different Learning Rates (3000 iterations)"
    estimator = LinearRegressionGD()
    plot_learning_curve(estimator, title, X_train_b, y_train, param_name='learning_rate', param_range=param_grid_lr, cv=3, n_jobs=-1)

    # -------------------------------------------

    plot_feature_influence(final_feature_names, best_model)















