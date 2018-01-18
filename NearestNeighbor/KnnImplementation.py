# encoding = utf-8
__author__ = 'young'
import csv
import random
import math
import operator

"""
将总数据集拆分成训练数据与测试数据两部份
1.filename样本数据文件
  split训练数据与测试数据拆分比例
  trainingSet训练数据集
  testSet测试数据集
2.循环样本数据集，生成一个随机值，如果小于split值归为训练数据集
  如果大于split值归为测试数据集
"""
def loadDataset(filename,split,trainingSet=[],testSet=[]):
    with open(filename,'r',encoding='utf-8') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            print("x:"+str(dataset[x+1]))
            for y in range(4):
                dataset[x+1][y] = float(dataset[x+1][y])
            if random.random() < split:
                trainingSet.append(dataset[x+1])
            else:
                testSet.append(dataset[x+1])

"""
计算两个实例之间的距离
1.将多维数据之间的距离累加
2.将两个实例之间距离累计值开方，得到实例之间的距离
"""
def euclideanDistance(instance1,instance2,length):
    distance = 0
    #数据维度，2D or 3D
    for x in range(length):
        #计算每个平面上两点之间的距离，并将每个平面的距离累加
        distance += pow((instance1[x] - instance2[x]),2)
    #返回开方值
    return math.sqrt(distance)

"""
获取k个距离最近的实例
1.遍历计算出所有训练数据与测试数据的距离
2.将计算出距离结果集排序
3.根据排序好的距离集取k个距离最近实例
"""
def getNeighbors(trainingSet,testInstance,k):
    distances = []
    length = len(testInstance)-1
    print("testInstance:"+str(testInstance))
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance,trainingSet[x],length)
        distances.append((trainingSet[x],dist))
    distances.sort(key = operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

"""
计算距离最近的几个实例分类进行投票，选取票数最多的类别名称作为结果
1.遍历k个实例类别名称
2.并分别存入classVotes对象
3.遍历时如果当前实例分类包含在classVotes对象时投票数加1，
  否则生成一个新的classVotes属性并将投票值设置为1
4.遍历投票结束后取投票数最多为分类结果
"""
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        # print("response:"+str(neighbors[x][3]))
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(),key = operator.itemgetter(1),reverse=True)
    # print("sotedVotes:"+str(sortedVotes[0][0]))
    return sortedVotes[0][0]
"""
根据预测数据集和预测结果集计算模型准确率
1.传入测试数据集和测试数据预测结果集
2.将这两个结果集进行比对，累加得到预测正确数值
3.预测正确数值/测试数据量
"""
def getAccuracy(testSet,predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

def main():
    trainingSet = []
    testSet = []
    split = 0.67
    loadDataset(r"C:\Users\yzh\PycharmProjects\deep-learning\NearestNeighbor\iris.csv",split,trainingSet,testSet)
    predictions = []
    k = 3
    print("testSetLen:"+str(len(testSet)))
    for x in range(len(testSet)):
        #取k个距离最近的实例
        neighbors = getNeighbors(trainingSet,testSet[x],k)
        # print("neighbors:"+str(neighbors))
        result = getResponse(neighbors)
        predictions.append(result)
        print('预测分类结果='+repr(result)+' ,测试数据实际分类结果='+repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet,predictions)
    print("Accuracy: "+repr(accuracy)+"%")

main()
#lear more link to http://www.yangzhihuiweb.com