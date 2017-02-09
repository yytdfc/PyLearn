import sys
import numpy as np
import matplotlib.pyplot as plt
# cluster class
class cluster:
    def __init__(self, idx, x, left=None, right=None):
        self._x=x          # cluster data
        self._idx=idx      # cluster index
        self._left=left    # left side of cluster
        self._right=right  # left side of cluster
        self._pltxy=None   # the [x,y] location in the hierarchical tree pictiure
    def idx(self):
        return self._idx
    def x(self):
        return self._x
    def setpltxy(self, pltxy):
        self._pltxy=np.array(pltxy,np.float32)
    def pltxy(self):
        return self._pltxy
    def right(self):
        return self._right
    def left(self):
        return self._left
# print each node of hierarchical tree
def printHC(cluster, pltIdx):
    if cluster.left()==None and cluster.right()==None:
        # arrange the location for the deepest node
        cluster.setpltxy([pltIdx,0])
        plt.text(pltIdx,-0.3,str(cluster.idx()),horizontalalignment='center',
                 verticalalignment='center',fontsize=1)
        pltIdx+=1
        return pltIdx
    else:
        # deep search left and right node
        pltIdx = printHC(cluster.left(), pltIdx)
        pltIdx = printHC(cluster.right(), pltIdx)
        return pltIdx
# get the file name
filename='cho.txt'
if len(sys.argv)>1:
    filename=sys.argv[1]
# load data
data=np.loadtxt(filename)
traindata=data[:,3:]
# initial the clusters
clusters=[cluster(idx=i, x=x) for i,x in enumerate(traindata)]
# distance of each cluster
distance={}
# total namber of data
N=len(clusters)
# store the new generate clusters
newclusters=[]
pltIdx=0
for n in range(N-1):
    minDis=np.inf
    for i in range(N-n):
        for j in range(i+1,N-n):
            # calculate the distance
            if (clusters[i].idx(),clusters[j].idx()) not in distance:
                distance[(clusters[i].idx(),clusters[j].idx())]=\
                np.dot(clusters[i].x()-clusters[j].x(),
                       clusters[i].x()-clusters[j].x())
            # find the nearest two clusters
            if distance[(clusters[i].idx(),clusters[j].idx())]<minDis:
               minDis=distance[(clusters[i].idx(),clusters[j].idx())]
               minI,minJ=i, j
    # generate a new cluster with two nearest clusters
    newcluster=cluster(x=(clusters[minI].x()+clusters[minJ].x())/2,
        idx=N+n, left=clusters[minI], right=clusters[minJ])
    # delete the two nearest clusters
    del clusters[minJ]
    del clusters[minI]
    # add the new cluster to clusters
    clusters.append(newcluster)
    newclusters.append(newcluster)
# print each node of hierarchical tree
printHC(clusters[0],0)
for cluster in newclusters:
    # get the left and right node location
    xI,yI=cluster.right().pltxy()
    xJ,yJ=cluster.left().pltxy()
    # draw the tree line
    cluster.setpltxy([(xI+xJ)/2,np.max([yI,yJ])+1])
    pltx=[xI,xI,xJ,xJ]
    plty=[yI,cluster.pltxy()[1],cluster.pltxy()[1],yJ]
    plt.plot(pltx,plty,'-',color = 'b')
# draw the top point
x0,y0=clusters[0].pltxy()
plt.plot([x0,x0],[y0,y0+1],'-',color = 'b')
# set the pictiure paras
plt.xlim(-1,N)
plt.ylim(-1,y0+1.5)
plt.axis('off')
# save the picture
plt.savefig('Hierarchical_tree.png',dpi=900)
