import numpy as np
import matplotlib.pyplot as plt
def load(filename):
    data = []
    fr = open(filename)
    for line in fr.readlines():
        linefloat = []
        for e in line.strip().split():
            linefloat.append(float(e))
        data.append(linefloat)
    return np.array(data)
def replaceNaN(filename):
    data = load(filename)
    for i in range(data.shape[1]):
        meanv = np.mean(data[np.nonzero(~np.isnan(data[:,i])), i])
        data[np.nonzero(np.isnan(data[:,i])), i] = meanv
    return data
def pca():
    filename = "pcadata/secom.data"
    data = replaceNaN(filename)
    m, n = data.shape
    pcafeature = 20
    meanv = np.tile(np.mean(data, axis=0).reshape(1,-1), (m,1))
    # covariance matrix = (x-u)T(x-u)
    covm = (data - meanv).T.dot(data - meanv)
    eigval, eigvec = np.linalg.eig(covm)
    eigindex = np.argsort(-eigval)
    eigind = eigindex[0:pcafeature]
    rEigvec = eigvec[:, eigind]
    # reduced_x = xp
    pcadata = data.dot(rEigvec)
    pcasum = np.sum(eigval[eigind])
    allsum = np.sum(eigval)
    print "pca: ",pcasum/allsum
    return pcadata
def pcatest():
    print "pca"
    pca()
if __name__ == "__main__":
    pcatest()
