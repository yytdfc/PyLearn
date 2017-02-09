# random_forrest.py
import numpy as np
import collections
import random
class Node():
    def __init__(self, dim, value, left=None, right=None, leftcls=None,rightcls=None):
        self.dim=dim
        self.divide_value=value
        self.left=left
        self.right=right
        self.leftcls=leftcls
        self.rightcls=rightcls
class RF():
    def __init__(self, train_input=None, train_output=None, testdata=None, N=3,NTree=3):
        self._NTree=NTree
        self._train_input = train_input
        self._train_output = train_output
        self._N=N
    def train(self, train_input=None, train_output=None):
        if train_input is None:
            print('mising train input!')
            return
        if train_output is None:
            print('mising train output!')
            return
        if len(train_input)!=len(train_output):
            print('train inputs\' and outputs\' are not euqal!')
            return
        self._train_input = np.array(train_input)
        self._train_output = np.array(train_output)
        self._dims=list(range(len(train_input[0])))
        self._roots=[]
        for i in range(self._NTree):
            self._dims=list(range(len(train_input[0])))
            self._roots.append(self.divide(self._train_input, self._train_output))
        self._class_dic={c:i for i,c in enumerate(set(self._train_output))}
        self._n_classes=len(self._class_dic)  
    def divide(self, train_input, train_output):
        dim=random.choice(self._dims)
        select_value=None
        select_entropy=np.inf
        mi,ma=np.min(train_input[:,dim]),np.max(train_input[:,dim])
        for v in np.linspace(mi,ma,self._N)[1:-1]:
            left=[y for x,y in zip(train_input,train_output) if x[dim]<=v]
            right=[y for x,y in zip(train_input,train_output) if x[dim]>v]
            n_left=len(left)
            xlog2x=lambda x:x*np.log2(x)
            leftEntropy=-n_left*np.sum([xlog2x(float(i)/n_left) 
                for i in collections.Counter(left).values()])
            n_right=len(right)
            rightEntropy=-n_right*np.sum([xlog2x(float(i)/n_left) 
                for i in collections.Counter(right).values()])
            entropy=leftEntropy+rightEntropy
            if entropy<select_entropy:
                select_value=v
                select_entropy=entropy
            if select_entropy==0:
                break
        if len(self._dims)==1:
            lefty=[y for x,y in zip(train_input,train_output) if x[dim]<=select_value]
            righty=[y for x,y in zip(train_input,train_output) if x[dim]>select_value]
            leftcls=collections.Counter(lefty).most_common(1)[0][0]
            if len(righty)==0:
                rightcls=leftcls
            else:
                rightcls=collections.Counter(righty).most_common(1)[0][0]
            node=Node(dim,select_value,leftcls=leftcls,rightcls=rightcls)
        else:
            leftx=np.array([x for x,y in zip(train_input,train_output) if x[dim]<=select_value])
            lefty=[y for x,y in zip(train_input,train_output) if x[dim]<=select_value]
            rightx=np.array([x for x,y in zip(train_input,train_output) if x[dim]>select_value])
            righty=[y for x,y in zip(train_input,train_output) if x[dim]>select_value]
            self._dims.remove(dim)
            if leftEntropy==0:
                leftcls=collections.Counter(lefty).most_common(1)[0][0]
                leftnode=None
            else:
                leftnode=self.divide(leftx,lefty)
                leftcls=None
            if rightEntropy==0:
                if len(righty)==0:
                    rightcls=leftcls
                else:
                    rightcls=collections.Counter(righty).most_common(1)[0][0]
                rightnode=None
            else:
                rightnode=self.divide(rightx,righty)
                rightcls=None
            node=Node(dim,select_value,leftnode,rightnode,leftcls,rightcls)
            self._dims.append(dim)
        return node
    def evaluate(self, x):
        cls=[]
        for node in self._roots:
            while 1:
                if x[node.dim]<=node.divide_value:
                    if node.left is None:
                        cls.append(node.leftcls)
                        break
                    node=node.left
                elif x[node.dim]>node.divide_value:
                    if node.right is None:
                        cls.append(node.rightcls)
                        break
                    node=node.right
        return collections.Counter(cls).most_common(1)[0][0]
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