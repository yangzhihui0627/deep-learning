# encoding = utf-8
__author__ = 'young'
from sklearn.feature_extraction import DictVectorizer
import csv
from numpy import array
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO

#Read in the csv file and put features in a list
allElectronicesData = open(r"C:\Users\yzh\PycharmProjects\deep-learning\DecisionTree\decisionTree.csv",'r',encoding='utf-8')
reader = csv.reader(allElectronicesData)
headers = reader.__next__()

#将特征向量与标签值提取出来
featureList = []
labelList = []
for row in reader:
    labelList.append(row[len(row) - 1])
    rowDict = {}
    for i in range(1,len(row) - 1):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)

#将特征向量转化成算法法识别的数据格式
vec = DictVectorizer()
print(featureList)
dummyX = vec.fit_transform(featureList).toarray()
# print("dummyX:"+str(dummyX))
# [[0. 0. 1. 0. 1. 1. 0. 0. 1. 0.]
#  [0. 0. 1. 1. 0. 1. 0. 0. 1. 0.]
#  [1. 0. 0. 0. 1. 1. 0. 0. 1. 0.]
#  [0. 1. 0. 0. 1. 0. 0. 1. 1. 0.]
#  [0. 1. 0. 0. 1. 0. 1. 0. 0. 1.]
#  [0. 1. 0. 1. 0. 0. 1. 0. 0. 1.]
#  [1. 0. 0. 1. 0. 0. 1. 0. 0. 1.]
#  [0. 0. 1. 0. 1. 0. 0. 1. 1. 0.]
#  [0. 0. 1. 0. 1. 0. 1. 0. 0. 1.]
#  [0. 1. 0. 0. 1. 0. 0. 1. 0. 1.]
#  [0. 0. 1. 1. 0. 0. 0. 1. 0. 1.]
#  [1. 0. 0. 1. 0. 0. 0. 1. 1. 0.]
#  [1. 0. 0. 0. 1. 1. 0. 0. 0. 1.]
#  [0. 1. 0. 1. 0. 0. 0. 1. 1. 0.]]

# 取特征值集合
# print(str(vec.get_feature_names()))
# [{'income': 'high', 'age': 'youth', 'credit_rating': 'fair', 'student': 'no'}, {'income': 'high', 'age': 'youth', 'credit_rating': 'excellent', 'student': 'no'}, {'income': 'high', 'age': 'middle_aged', 'credit_rating': 'fair', 'student': 'no'}, {'income': 'medium', 'age': 'senior', 'credit_rating': 'fair', 'student': 'no'}, {'income': 'low', 'age': 'senior', 'credit_rating': 'fair', 'student': 'yes'}, {'income': 'low', 'age': 'senior', 'credit_rating': 'excellent', 'student': 'yes'}, {'income': 'low', 'age': 'middle_aged', 'credit_rating': 'excellent', 'student': 'yes'}, {'income': 'medium', 'age': 'youth', 'credit_rating': 'fair', 'student': 'no'}, {'income': 'low', 'age': 'youth', 'credit_rating': 'fair', 'student': 'yes'}, {'income': 'medium', 'age': 'senior', 'credit_rating': 'fair', 'student': 'yes'}, {'income': 'medium', 'age': 'youth', 'credit_rating': 'excellent', 'student': 'yes'}, {'income': 'medium', 'age': 'middle_aged', 'credit_rating': 'excellent', 'student': 'no'}, {'income': 'high', 'age': 'middle_aged', 'credit_rating': 'fair', 'student': 'yes'}, {'income': 'medium', 'age': 'senior', 'credit_rating': 'excellent', 'student': 'no'}]
# ['age=middle_aged', 'age=senior', 'age=youth', 'credit_rating=excellent', 'credit_rating=fair', 'income=high', 'income=low', 'income=medium', 'student=no', 'student=yes']

#将classLable处理成所识别的程序格式
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
# print("dummyY:"+str(dummyY))

#添加分类器对象，设置相关算法(ID3-信息熵)，开始训练数据模型
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX,dummyY)

# 分类器配置信息
# print("clf:"+str(clf))
# clf:DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
#             max_features=None, max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, presort=False, random_state=None,
#             splitter='best')

#将决策树另存为dot文件
with open("allElectronicInformationGainOri.dot","w") as f:
    f = tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file=f)


#传入测试数据，查看预测结果
testRowX = {'credit_rating': 'excellent', 'income': 'high', 'age': 'youth', 'student': 'no'}
#转化测试数据格式
testFeatures = array(vec.transform(testRowX).toarray()).reshape(1,-1)
# print("testFeatures;"+str(testFeatures))
# testFeatures;[[0. 0. 1. 1. 0. 1. 0. 0. 1. 0.]]
# predictedY = clf.predict(array(testRowX).reshape(1, -1))
predictedY = clf.predict(testFeatures)
#输出预测结果
print(predictedY)
# [0]
#学习更多相关内容请访问站点http://www.yangzhihuiweb.com