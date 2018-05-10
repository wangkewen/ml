import numpy as np
import matplotlib.pyplot as plt
def loaddata(filename):
    listdata = []
    label = []
    filep = open(filename,"r")
    for line in filep.readlines():
        linearr = line.strip().split("\t")
        listdata.append([float(linearr[0]), float(linearr[1]), 1.0])
        label.append([int(linearr[2])])
    return np.array(listdata), np.array(label)
def fig(data):
    x = range(len(data))
    plt.plot(x,data)
    plt.show()
def logreg():
    filename = "logdata/TestSet.txt"
    data, label = loaddata(filename)
    itern = 1000
    m, n = data.shape
    alpha = 0.001
    loss = 1
    weight = np.ones((n, 1))
    lossf = []
    for s in range(itern):
        # h(x) = xw
        # sig = 1 / 1 + exp(-h(x))
        sig = 1.0 / (1 + np.exp(-data.dot(weight)))
        # loss = -ylog(h(x)) - (1-y)log(1-h(x))
        loss = -label.T.dot(np.log(sig)) - (1 - label).T.dot(np.log(1 - sig))
        lossf.append(loss[0])
        # gradient = x(y-sig)
        gradient = data.T.dot(label - sig)
        weight = weight + alpha * gradient
    #fig(lossf)
    return data, label, weight
def testlog():
    print "log"
    data, label, weight = logreg()
    hit = 0.0
    hitc = 0.0
    testc = 0
    predrs = 1 / (1 + np.exp(-data.dot(weight)))
    m, n = label.shape
    for i in range(m):
        testc += label[i][0]==1
        if predrs[i][0]>=0.5:
            hit += label[i][0]==1
            hitc += label[i][0]==1
        else:
            hit += label[i][0]==0
    print "Pred Accuracy ",hit/m
    print "Recall rate ", hitc/testc
if __name__=="__main__":
    testlog()
