from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import GridSearchCV

train = pd.read_csv('adult_train.csv', header=None, index_col=False,
                   names=['年龄','单位性质','统计权重','学历',
                          '受教育时长','婚姻状况','职业','家庭状况',
                          '种族','性别','资产现状','负债',
                          '周工作时长','原国籍','年收入'])
test = pd.read_csv('adult_test.csv', header=None, index_col=False,
                   names=['年龄','单位性质','统计权重','学历',
                          '受教育时长','婚姻状况','职业','家庭状况',
                          '种族','性别','资产现状','负债',
                          '周工作时长','原国籍','年收入'])
print(train.shape)
print(test.shape)
# 选取我们感兴趣的几个特征
train_select = train[['年龄','单位性质','学历','性别','职业','年收入']]
test_select = test[['年龄','单位性质','学历','性别','职业','年收入']]
print(train_select.head())
print('=============================================================================================\n')
# 对特征进行独热码编码转换
train_dummies = pd.get_dummies(train_select)
test_dummies = pd.get_dummies(test_select)
# 显示全部列
pd.set_option('display.max_columns', None)
print(train_dummies.head())
print('=============================================================================================\n')
train_features = train_dummies.loc[:,'年龄':'职业_ Transport-moving']
test_features = test_dummies.loc[:,'年龄':'职业_ Transport-moving']

X = train_features.values
Y = train_dummies['年收入_ >50K'].values
x = test_features.values
y = test_dummies['年收入_ >50K'].values

print(X.shape)
print(Y.shape)
print(x.shape)
print(y.shape)

# 将数据集拆分为训练集和测试集
# X_train,X_test,Y_train,Y_test = train_test_split(X, Y, random_state=0)

# 设置分类器为随机森林
Rf = RandomForestClassifier()
# 定义随机森林的各种参数
parameter = {'n_estimators': [50,100,200],             # 子模型的个数
              'criterion': ['entropy'],        # 判断节点是否继续分裂采用的计算方法
              'max_depth': [4,5,6],                  # 随机森林的最大深度
              'min_samples_split': [2, 6,8],          # 分裂所需的最小样本数
              'min_samples_leaf': [2, 6, 10],        # 叶节点最小样本数
              'max_features': [5 ,20,25,30 ,43],  # 节点分裂时参与判断的最大特征数
              'bootstrap': [True]     # 是否bootstrap对样本抽样
              }
# 自动调整参数，通过交叉验证确定最佳参数
grid_obj = GridSearchCV(Rf,parameter,cv=10)
grid_obj = grid_obj.fit(X,Y)

# 相亲对象 = [['24',   # 年龄
# # 国家公务员、市县公务员、无业、私企、自由职业、个体户、省直公务员
#         '0','0','1','0','0', '0','0','0','0',
# # 高一、高二、高三、小学没毕业、小学毕业、初中没毕业、初中毕业
#         '0','0','0','0','0','0','0',
# # Assoc-acdm、assoc-voc、本科、博士、高中毕业、硕士、幼儿园、profschool、上过大学
#         '0','0','1','0','0','0','0','0','0', # 学历
# # 女性、男性
#         '0','1',
# # 未知、文员、武警、修理工、高级管理人员、渔夫、清洁工、设备员、服务行业、私人服务业、教授、保安、销售、工程技术人员、运输业、
#         '0','0','0','0','0','0','0','0','0','0','0','0','0','1','0',]]
#
#
# dating_decide = Rf.predict(相亲对象)

# 获取最佳模型
clf = grid_obj.best_estimator_
clf.fit(X,Y)

print('=============================================================================================\n')
print('模型分类最佳准确度：{:.2f}'.format(grid_obj.best_score_))
print('模型分类最佳参数：{}'.format(grid_obj.best_params_))


# print('=============================================================================================\n')
# if dating_decide == 1:
#     print("这哥们买房指日可待啊")
# else:
#     print("别去了，这哥们供不起厦门一套房")