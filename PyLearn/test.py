import numpy as np
import knn
import navie_bayes as nb
import decision_tree as dt
import random_forrest as rf
import boosting as bt
print('Loading data.txt...')
data=np.loadtxt('data.txt')
trainin=data[:,:-1]
trainout=data[:,-1]
print('classify with KNN')
knn1=knn.KNN(k=3)
knn1.train(trainin,trainout)
knn1.test(cross_fold=10)
print('\n')
print('classify with Navie Bayes')
nb1=nb.NB()
nb1.train(trainin,trainout)
nb1.test(cross_fold=10)
print('\n')
print('classify with Decision Tree')
dt1=dt.DT(N=5)
dt1.train(trainin,trainout)
dt1.test(cross_fold=10)
print('\n')
print('classify with Random Forrest')
rf1=rf.RF(N=5,NTree=5)
rf1.train(trainin,trainout)
rf1.test(cross_fold=10)
print('\n')
print('classify with Boosting')
trainin=data[:,:-1]
trainout=data[:,-1]
bt1=bt.Boosting()
bt1.train(trainin,trainout)
bt1.test(cross_fold=10)
print('\n\n')
