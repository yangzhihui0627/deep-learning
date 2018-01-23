# encoding = utf-8
__author__ = 'young'
import numpy as np
from matplotlib import pylab
from sklearn import  svm

# x = [[2,0],[1,1],[2,3]]
# y = [0,0,1]
# clf = svm.SVC(kernel='linear')
# clf.fit(x,y)
# print("clf:"+str(clf))
# print("support_vectors:"+str(clf.support_vectors_))
# print("n_support:"+str(clf.n_support_))

#seed值设置为0，确保每次生成的值类别一致
np.random.seed(0)
#生成40行2列矩阵数据，其中20行同时减2偏移，另20行统一加2偏移
X = np.r_[np.random.randn(20,2)-[2,2],np.random.randn(20,2)+[2.2]]
Y = [0]*20 + [1]*20

#训练生成模型
clf = svm.SVC(kernel="linear")
clf.fit(X,Y)

#计算得到超平面
# w0x0 + w1x1 + w3 = 0 => y= -(w0/w1)x + (w3/w1)
# w0/w1得到斜率, w3/w1得到截距
w=clf.coef_[0]
a=-w[0]/w[1]
#生成一批-5到5的连续的float类型的数值
xx=np.linspace(-5,5)
yy=a*xx-(clf.intercept_[0]/w[1])

#通过计算得到支持向量上的两条斜线，由于这两个斜线与超平面平行，因此可以利用上边已经算出来的斜率值a与x值xx
b = clf.support_vectors_[0]
#y=ax+b => b=y-ax
yy_down = a*xx +(b[1] - a*b[0])
#取另一个分类的支持向量
b = clf.support_vectors_[-1]
yy_up = a*xx + (b[1] - a*b[0])

#根据得到三个直线方程画线
pylab.plot(xx,yy,'k-')
pylab.plot(xx,yy_down,'k--')
pylab.plot(xx,yy_up,'k--')

pylab.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],s=80,facecolors='none')
pylab.scatter(X[:,0],X[:,1],c=Y,cmap="prism")
pylab.axis("tight")

pylab.show()