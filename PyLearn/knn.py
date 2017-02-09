# knn.py
import numpy as np
import heapq
import collections
import random
class KNN():
    def __init__(self, train_input=None, train_output=None, testdata=None, k=1):
        self._train_input = train_input
        self._train_output = train_output
        if k%2==0: k-=1
        if k<=0:k=1
        self._k=k
    def set_paras(self, k=1, distance_fun='Euclidean'):
        if k%2==0: k-=1
        if k<=0:k=1
        self._k=k
        self._distance_fun=distance_fun
    def train(self, train_input=None, train_output=None):
        self._train_input = np.array(train_input)
        self._train_output = np.array(train_output)
        self._class_dic={c:i for i,c in enumerate(set(self._train_output))}
        self._n_classes=len(self._class_dic)
    def evaluate(self, x):
        dis_fun=lambda y:np.sum(np.abs(y-x))
        idxs=findLeastK([dis_fun(i) for i in self._train_input],self._k)
        return collections.Counter([self._train_output[idx] for idx in idxs]).most_common(1)[0][0]
    def test(self, test_input=None, test_output=None, cross_fold=1):
        if not(test_input is None or test_output is None):
            evaluate_test=[self.evaluate(x) for x in test_input]
        if cross_fold>1:
            test_input=self._train_input
            test_output=self._train_output
            n_train=len(test_input)
            idxs=list(range(n_train))
            random.shuffle(idxs)
            evaluate_test=[None for i in range(n_train)]
            fold_n=int(n_train/cross_fold)
            if n_train%cross_fold!=0:fold_n+=1
            for i in range(cross_fold):
                train_idxs=idxs[:i*fold_n]+idxs[(i+1)*fold_n:]
                test_idxs=idxs[i*fold_n:(i+1)*fold_n]
                self.train(test_input[train_idxs],[test_output[j] for j in train_idxs])
                for j in test_idxs:
                    evaluate_test[j]=self.evaluate(test_input[j])
            self._train_input=test_input
            self._train_output=test_output
        confusion_matrix=np.zeros((self._n_classes,self._n_classes),dtype=np.uint64)
        for i,j in zip(test_output,evaluate_test):
            confusion_matrix[self._class_dic[i],self._class_dic[j]]+=1
        accuracy=float(np.sum([confusion_matrix[i,i] for i in range(self._n_classes)]))/len(test_input)   
        print('confusion matrix:')
        for i,row in enumerate(confusion_matrix):
            print(str(self._class_dic[i])+':\t'+str(row))
#        print(confusion_matrix)
        print('Accuracy = '+str(accuracy))
        for i,row in enumerate(confusion_matrix):
            print('Pression('+str(self._class_dic[i])+') = '+str(float(confusion_matrix[i,i])/np.sum(confusion_matrix[:,i])))
            print('Recall('+str(self._class_dic[i])+')  = '+str(float(confusion_matrix[i,i])/np.sum(confusion_matrix[i,:])))
            print('F-mesure('+str(self._class_dic[i])+')  = '+str(float(confusion_matrix[i,i])/(np.sum(confusion_matrix[:,i])+np.sum(confusion_matrix[i,:])-confusion_matrix[i,i])))
def findLeastK(x, k):
    length=len(x)
    maxheap=[]
    if not x or k<=0 or k>length:
        return
    for i,idx in zip(x[:k],range(k)):
        heapq.heappush(maxheap,(-i,idx))
    for i,idx in zip(x[k:],range(k,length)):
        heapq.heappushpop(maxheap,(-i,idx))
    return [i[1] for i in maxheap]