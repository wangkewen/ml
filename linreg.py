import numpy as np
import matplotlib.pyplot as plt
import math
import random
def load(filename):
    filep = open(filename)
    data = []
    label = []
    for line in filep.readlines():
        linearr = line.strip().split(",")
        oneline = []
        label.append([float(linearr[0])])
        for i in range(len(linearr)-1):
            oneline.append(float(linearr[i+1]))
        #oneline.append(1.0)
        data.append(oneline)
        #label.append([float(linearr[-1])])
    return np.array(data), np.array(label)
def norm(data):
    m, n = data.shape
    vmean = np.mean(data,axis=0).reshape(-1,1).T
    vstd = np.std(data,axis=0).reshape(-1,1).T
    normdata = data - np.tile(vmean, (m,1))
    normdata = normdata / np.tile(vstd, (m,1))
    return normdata
def linreg():
    filename = "lindata/wine.data"
    adata, alabel = load(filename)
    adata = norm(adata)
    alabel = norm(alabel)
    itern = 3000
    trainratio = 0.9
    trainn = int(math.floor(trainratio*adata.shape[0]))
    data = adata[0:trainn]
    label = alabel[0:trainn]
    m,n = data.shape
    weight = np.zeros((n,1))
    alpha = 0.3
    lambdar = 0.01
    lossf = []
    is_stochastic = False
    is_ridge = True
    for i in range(itern):
        # h(x) = xw
        # loss = (1/2m)(y-h(x))^2
        loss = np.sum((label - data.dot(weight)) ** 2)/(2*m)
        # ridgeloss = (1/2m)(y-h(x))^2 + (lambda/2)w^2
        if is_ridge:
            loss = loss + lambdar * np.sum(weight ** 2)/2
        lossf.append(loss)
        # gradient = x(y-h(x))/m
        if not is_stochastic:
            gradient =  data.T.dot(label - data.dot(weight))/m
        # stochastic gradient descent
        else:
            j = random.randint(0,m-1)
            onedata = data[j].reshape(1,-1)
            onelabel = label[j].reshape(1,-1)
            gradient = onedata.T.dot(onelabel - onedata.dot(weight))/m
        if is_ridge:
            gradient = gradient + lambdar * weight
        weight = weight + alpha * gradient
    #x = range(itern)
    #plt.plot(x,lossf)
    #plt.show()
    return adata[trainn:-1], alabel[trainn:-1], weight
def regtest():
    print "linreg"
    testdata, testlabel, weight = linreg()
    m, n = testdata.shape
    mse = np.sum((testlabel - testdata.dot(weight)) ** 2)/m
    print "mean square error: ",mse
if __name__ == "__main__":
    regtest()
