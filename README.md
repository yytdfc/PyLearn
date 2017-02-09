# PyLearn
This is a simple implement of usual machine learing algorithm.

### Classifier:
- KNN, Navie Bayes, Decision Tree, Random Forrest, ~~ANN~~

### Clustering:
- Hierarchical Clustering, Density Based Clustering, ~~KMeans~~

***To be comtinued***

### Example:
This is a simple example of KNN, Navie Bayes and Decision Tree, other modules goes the same way:

```python
import numpy as np
import knn
import navie_bayes as nb
import decision_tree as dt
data=np.loadtxt('data.txt')
trainin=data[:,:-1]
trainout=data[:,-1]
print('classify with KNN')
knn1=knn.KNN(k=3)
knn1.train(trainin,trainout)
knn1.test(cross_fold=10)
print('\nclassify with Navie Bayes')
nb1=nb.NB()
nb1.train(trainin,trainout)
nb1.test(cross_fold=10)
print('\nclassify with Decision Tree')
dt1=dt.DT(N=5)
dt1.train(trainin,trainout)
dt1.test(cross_fold=10)
```

result:
```
classify with KNN
confusion matrix:
0:	[346  11]
1:	[ 26 186]
Accuracy = 0.9349736379613357
Pression(0) = 0.930107526882
Recall(0)  = 0.96918767507
F-mesure(0)  = 0.903394255875
Pression(1) = 0.944162436548
Recall(1)  = 0.877358490566
F-mesure(1)  = 0.834080717489

classify with Navie Bayes
confusion matrix:
0:	[352   5]
1:	[ 35 177]
Accuracy = 0.929701230228471
Pression(0) = 0.909560723514
Recall(0)  = 0.985994397759
F-mesure(0)  = 0.897959183673
Pression(1) = 0.972527472527
Recall(1)  = 0.834905660377
F-mesure(1)  = 0.815668202765

classify with Decision Tree
confusion matrix:
0:	[336  21]
1:	[ 44 168]
Accuracy = 0.8857644991212654
Pression(0) = 0.884210526316
Recall(0)  = 0.941176470588
F-mesure(0)  = 0.837905236908
Pression(1) = 0.888888888889
Recall(1)  = 0.792452830189
F-mesure(1)  = 0.721030042918
```
