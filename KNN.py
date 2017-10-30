from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [1, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 0.5
    sqDistance = sqDiffMat.sum(axis=1)
    distances = sqDistance ** 0.5
    sortedDistance = distances.argsort()
    classcount = {}
    for i in range(k):
        votelLabel = labels[sortedDistance[i]]
        classcount[votelLabel] = classcount.get(votelLabel, 0) + 1
    sortedClasscount = sorted(classcount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClasscount[0][0]
