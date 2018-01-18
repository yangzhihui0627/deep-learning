# encoding = utf-8
__author__ = 'young'
import math

def ComputeEuclideanDistance(x1,y1,x2,y2):
    #求两点之间的距离
    d = math.sqrt(math.pow((x1-x2),2) + math.pow((y1-y2),2))
    return d

#计算距离
d_ag = ComputeEuclideanDistance(3,104,18,90)
d_bg = ComputeEuclideanDistance(2,100,18,90)
d_cg = ComputeEuclideanDistance(1,81,18,90)
d_dg = ComputeEuclideanDistance(101,10,18,90)
d_eg = ComputeEuclideanDistance(99,5,18,90)
d_fg = ComputeEuclideanDistance(98,2,18,90)

print("d_ag:"+str(d_ag))
print("d_bg:"+str(d_bg))
print("d_cg:"+str(d_cg))
print("d_dg:"+str(d_dg))
print("d_eg:"+str(d_eg))
print("d_fg:"+str(d_fg))

# d_ag:20.518284528683193
# d_bg:18.867962264113206
# d_cg:19.235384061671343
# d_dg:115.27792503337315
# d_eg:117.41379816699569
# d_fg:118.92854997854805