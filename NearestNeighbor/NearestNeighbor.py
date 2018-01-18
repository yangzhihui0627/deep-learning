# encoding = utf-8
__author__ = 'young'
from sklearn import neighbors
from sklearn import datasets

#创建knn算法实例
knn = neighbors.KNeighborsClassifier
iris = datasets.load_iris()
# print(iris)
knn.fit(iris.data,iris.target)
predicetedLabel = knn.predict([4.9, 3. , 1.4, 0.2])
print("result:"+predicetedLabel)