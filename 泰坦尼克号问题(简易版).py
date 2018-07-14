
# coding: utf-8

# In[1]:


#加载预处理数据将会用到的库
import pandas as pd
import seaborn as sns
from scipy import stats,integrate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# In[2]:


# 加载数据
datas = pd.read_csv('datas.csv')
datas.head()


# 从上述数据中可以发现PassengerId为ID自增列数据，属于无用数据

# In[3]:


datas.info()


# In[4]:


datas.describe()


# 从上述数据统计中可以发现在Age， Embarked中存在缺失值

# In[5]:


# 移除无用列 passengerId
datas.drop(['PassengerId'], axis=1, inplace=True)
datas.head()


# In[6]:


#处理缺失值
    #Embarked
        #查看Embarked各数据出现次数
sns.countplot(datas.Embarked.dropna())
datas.Embarked.value_counts()


# 从数据出现次数来看，Embarked为离散型数据，且绝大多数出现都为0，采用众数0填充

# In[7]:


datas.Embarked.fillna(0, inplace=True)


# In[8]:


#Age
#查看Age各数据出现次数
datas.Age.value_counts()
sns.distplot(datas.Age.dropna(), kde=True, bins=30, fit=stats.gamma)


# 从数据出现次数来看，Age连续数据，可采用预测填充

# In[9]:


# 首先对数据进行抽取，将数据分为年龄已知数据和年龄未知数据
datas_age = datas[[ 'Age', 'Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']].copy()
known_age_datas = datas_age.loc[datas_age.Age.notnull()].as_matrix()
unknow_age_datas = datas_age.loc[datas_age.Age.isnull()].as_matrix()
#梳理年龄预测用数据
x = known_age_datas[:, 1:]
x_test = unknow_age_datas[:, 1:]
y = known_age_datas[:, 0]
#采用随机森林进行预测
rfr = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
rfr.fit(x, y)
pred = rfr.predict(x_test)
datas.loc[datas.Age.isnull(), 'Age'] = pred


# In[10]:


datas.describe()


# In[11]:


#导入绘制图形将会用到的库
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


# In[12]:


#绘制图形函数


#绘制混淆矩阵图
def plot_confusion_matrix(model_name, conf_mat):
    #定义热图颜色
    ddl_heat = ['#0066FF', '#247CFF',  '#3E8BFF', '#4D94FF', '#5B9DFF', '#6EA8FF',
                '#7CB1FF', '#8EBCFF', '#9EC5FF', '#C0DAFF', '#DFECFF', '#E8F1FF']
    ddlheatmap = colors.ListedColormap(ddl_heat)
    # 绘制热图
    plt.imshow(conf_mat, interpolation='nearest', cmap=ddlheatmap)
    plt.title('{model_name} Confusion Matrix'.format(model_name=model_name))
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

#绘制PR图
def plot_precision_recall(model_name, precision, recall):
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title( '{model_name} 2-class Precision-Recall'.format(model_name=model_name))
    plt.show()


# In[13]:


#导入评估分析模型要用到的库
from sklearn import metrics


# In[14]:


#模型评估函数
#模型评估
def model_evaluation(train_true, train_pred, test_true, test_pred, pred_score):
    train_score = metrics.accuracy_score(train_true, train_pred)
    test_score = metrics.accuracy_score(test_true, test_pred)
    mse = metrics.mean_squared_error(test_true, test_pred)
    r2score = metrics.r2_score(test_true, test_pred)
    conf_mat = metrics.confusion_matrix(test_true, test_pred)
    precision, recall, _ = metrics.precision_recall_curve(test_true, pred_score)
    return train_score, test_score, mse, r2score, conf_mat, precision, recall

#模型分析
def auto_model_analysis(model_name, train_true, train_pred, test_true, test_pred, pred_score):
    train_score, test_score, mse, r2score, conf_mat, precision, recall = model_evaluation(train_true, train_pred,
                                                                                          test_true, test_pred,
                                                                                          pred_score)

    print("{model_name}模型训练集准确率：{train_score}".format(model_name=model_name, train_score=train_score))
    print("{model_name}模型验证集准确率：{test_score}".format(model_name=model_name, test_score=test_score))
    print("{model_name}模型均方误差：{mse}".format(model_name=model_name, mse=mse))
    print("{model_name}模型R2-score：{r2score}".format(model_name=model_name, r2score=r2score))
    print("{model_name}模型混淆矩阵：{conf_mat}".format(model_name=model_name, conf_mat=conf_mat))

    plot_confusion_matrix(model_name, conf_mat)
    plot_precision_recall(model_name, precision, recall)


# In[15]:


a = ['Pclass', 'Sex', 'Age', 'Parch', 'Fare']
# 获取列名
columns = datas.columns.values.tolist()
columns_label = columns[0:1]
columns_train = columns[1:]
scaler = StandardScaler()
datas_train = datas.copy()
datas_train.Age= scaler.fit_transform(datas_train.Age.values.reshape(-1, 1))
datas_train.Fare = scaler.fit_transform(datas_train.Fare.values.reshape(-1, 1))
X_train, X_test, Y_train, Y_test = train_test_split(datas_train[columns_train].as_matrix(), 
                                                    datas_train[columns_label].as_matrix().reshape(-1), 
                                                    test_size=0.3, random_state=0)


# In[16]:


#对数几率回归模型
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs', C=0.1, multi_class='ovr', penalty='l2', max_iter=5000)
lr.fit(X_train, Y_train)
train_pred = lr.predict(X_train)
test_pred = lr.predict(X_test)
pre_score = lr.predict_proba(X_test)[:, 1]
print("对数几率回归模型参数：{parameter}".format(parameter=lr))
auto_model_analysis("logistic_regression", Y_train, train_pred, Y_test, test_pred, pre_score)


# In[17]:


#决策树
from sklearn import tree
tree = tree.DecisionTreeClassifier(criterion="entropy", splitter='best', max_depth=4)
tree.fit(X_train, Y_train)
train_pred = tree.predict(X_train)
test_pred = tree.predict(X_test)
pre_score = tree.predict_proba(X_test)[:, 1]
print("决策树模型参数：{parameter}".format(parameter=tree))
auto_model_analysis("decision tree", Y_train, train_pred, Y_test, test_pred, pre_score)


# In[18]:


#神经网络
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='sgd', activation='tanh', hidden_layer_sizes=(7, 28, 56, 2), 
                    alpha=0.05, learning_rate_init=0.02, max_iter=10000)
mlp.fit(X_train, Y_train)
train_pred = mlp.predict(X_train)
test_pred = mlp.predict(X_test)
pre_score = mlp.predict_proba(X_test)[:, 1]
print("神经网络学习率：{rate}".format(rate=mlp.learning_rate_init))
print("神经网络正则化：{aplha}".format(aplha=mlp.alpha))
print("神经网络模型参数：{parameter}".format(parameter=mlp))
auto_model_analysis("neural network", Y_train, train_pred, Y_test, test_pred, pre_score)


# In[19]:


#支持向量机
from sklearn import svm
svc = svm.SVC(kernel='rbf', C=6, decision_function_shape='ovr', probability=True, max_iter=5000)
svc.fit(X_train, Y_train)
train_pred = svc.predict(X_train)
test_pred = svc.predict(X_test)
pre_score = svc.predict_proba(X_test)[:, 1]
y_test_score = svc.decision_function(X_test)
print("支持向量机模型参数：{parameter}".format(parameter=svc))
auto_model_analysis("svc", Y_train, train_pred, Y_test, test_pred, pre_score)

