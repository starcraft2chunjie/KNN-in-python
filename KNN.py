from numpy import *
import operator

def createDataSet():     #create the train dataset
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [1, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):    #the function of classify the inX into the correct category
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet   #repeat the test dataset in column
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distances = sqDistance ** 0.5    #compute the distance
    sortedDistance = distances.argsort()   #return the order of the index of the traindataset
    classcount = {}
    for i in range(k):
        votelLabel = labels[sortedDistance[i]]    #get the label of dataset no more than k
        classcount[votelLabel] = classcount.get(votelLabel, 0) + 1   #compute the num of appear for each label
    sortedClasscount = sorted(classcount.items(), key = operator.itemgetter(1), reverse = True)    #sort on the base of the tuple of classcount, and you should sort it by evaluating the second part of each tuple
    return sortedClasscount[0][0]

