import sys
import numpy as np
# get the file name
filename='cho.txt'
if len(sys.argv)>1:
    filename=sys.argv[1]
# load data
data=np.loadtxt(filename)
traindata=data[:,3:]
# set MintPts
MinPts=4
# set MinPts
Eps=0.98
# total namber of data
N=len(traindata)
# calculate the Eps*Eps for quick compare
Eps=Eps*Eps
# stroe each point's reachable neighbor
neighbor={}
for i in range(N):
    for j in range(i+1,N):
        if np.dot(traindata[i]-traindata[j],traindata[i]-traindata[j])<Eps:
            if i in neighbor:
                neighbor[i].add(j)
            else:
                neighbor[i]={j}
            if j in neighbor:
                neighbor[j].add(i)
            else:
                neighbor[j]={i}
# get each point's reachable number
densityReachable={i for i in neighbor if len(neighbor[i])>=MinPts}
# store the cluster
clusters=[]
while(densityReachable):
    k=densityReachable.pop()
    # generate a new cluster
    cluster={k}
    # a BFS search
    queue=[k]
    while(queue):
        idx=queue.pop(0)
        for i in list(neighbor[idx]):
            if i in densityReachable:
                cluster.add(i)
                queue.append(i)
                # remove the points already clustered
                neighbor[idx].remove(i)
                neighbor[i].remove(idx)
                if i in densityReachable: densityReachable.remove(i)
    # append cluster to clusters
    clusters.append(cluster)
print('Cluster into '+str(len(clusters))+' clusters.')
print('Each cluster size:')
print([len(i) for i in clusters])
# write result to density_based_result.txt
with open('density_based_result.txt','w') as f:
    for i,cluster in enumerate(clusters):
        f.write('Cluster '+str(i)+':\n')
        for k in cluster:
            f.write(str(k+1)+' ')
        f.write('\n')
