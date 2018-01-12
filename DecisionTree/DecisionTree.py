# encoding = utf-8
__author__ = 'young'
import os
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
print("dummyX:"+str(dummyX))
# dummyY = vec.fit_transform(labelList).toarray()
# print(str(vec.get_feature_names()))

#将classLable处理成所识别的程序格式
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
# print("dummyY:"+str(dummyY))

#添加分类器对象，设置相关算法(ID3-信息熵)，开始训练数据模型
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX,dummyY)
print("clf:"+str(clf))
#将决策树另存为dot文件
with open("allElectronicInformationGainOri.dot","w") as f:
    f = tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file=f)


#传入测试数据，查看预测结果
oneRowX = dummyX[0, :]
newRowX = oneRowX
print("new :"+str(newRowX))
# newRowX[0] = 1
print("newRowX:"+str(newRowX))
# predictedY = clf.predict(array(newRowX).reshape(1, -1))
predictedY = clf.predict(array(vec.transform({'credit_rating': 'excellent', 'income': 'high', 'age': 'youth', 'student': 'no'}).toarray()).reshape(1,-1))
print(predictedY)