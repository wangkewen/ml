import os
import numpy as np
import operator
def to2darray(c,filedirstr):
    filedir = os.listdir(filedirstr)
    r = len(filedir)
    label = []
    resultdata = np.zeros((r,c))
    for l in range(r):
        filename = filedir[l].split(".")[0]
        label.append(int(filename.split("_")[0]))
        filep = open(filedirstr+"/"+filedir[l],"r")
        for i in range(32):
            line = filep.readline()
            for j in range(32):
                resultdata[l, i*32+j] = int(line[j])
    return resultdata, label
def loadfile():
    trainfiledir = "knndata/trainingDigits"
    testfiledir = "knndata/testDigits"
    traindata, trainlabel = to2darray(1024,trainfiledir)
    testdata, testlabel = to2darray(1024,testfiledir)
    return traindata, trainlabel, testdata, testlabel
def knn(traindata,trainlabel,test,k):
    r,c = traindata.shape
    diff = np.tile(test, (r,1)) - traindata
    distance = (diff ** 2).sum(axis=1) ** 0.5
    sortIndex = distance.argsort()
    groupCount = {}
    for i in range(k):
        onelabel = trainlabel[sortIndex[i]]
        groupCount[onelabel] = groupCount.get(onelabel, 0) + 1
    resultlabel = sorted(groupCount.items(), key=operator.itemgetter(1), \
                            reverse=True)
    return resultlabel[0][0]
def knntest():
    traindata,trainlabel,testdata,testlabel=loadfile()
    hit = 0
    n = len(testlabel)
    k = 3
    labeldis = {}
    for i in range(n):
        pred = knn(traindata,trainlabel,testdata[i],k)
        hit += (pred == testlabel[i])
        labeldis[testlabel[i]] = labeldis.get(testlabel[i], 0) + (1.0/n)
    for i, freq in labeldis.items():
        print(i," ",freq)
    print("Prediction Accuracy ", 1.0*hit/n)
if __name__ == '__main__':
    knntest()
