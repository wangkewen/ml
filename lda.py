import numpy as np
import matplotlib.pyplot as plt
def load(filename):
    data = []
    label = []
    fr = open(filename)
    for line in fr.readlines():
        linearr = line.strip().split(",")
        oneline = []
        label.append(int(linearr[0]))
        for i in range(len(linearr)-1):
            oneline.append(float(linearr[i+1]))
        data.append(oneline)
    return np.array(data), label
def lda():
    filename = "lindata/wine.data"
    data, label = load(filename)
    m, n = data.shape
    ldafeature = 3
    # count of each class#
    labelc = np.bincount(label)
    # accumulate sum
    labelcum = np.cumsum(labelc)
    # mean of data
    xmean = np.mean(data, axis=0).reshape(1,-1)
    sw = np.zeros((n,n))
    sb = np.zeros((n,n))
    for i in range(len(labelc)):
        if labelc[i] != 0:
            xi = data[labelcum[i-1]:labelcum[i],:]
            # mean of class i
            ximean = np.tile(np.mean(xi, axis=0).reshape(1,-1), (labelc[i],1))
            # Within-class scatter matrix Sw
            # sw = sum(x-mi).T(x-mi)
            sw = sw + (xi-ximean).T.dot(xi-ximean)
            cxmean = np.tile(xmean, (labelc[i],1))
            # Between-class scatter matrix
            # sb = sum(n(mi-m).T(mi-m)
            sb = sb + labelc[i]*(ximean - cxmean).T.dot(ximean - cxmean)
    # sb/sw
    eigval, eigvec = np.linalg.eig(np.linalg.inv(sw).dot(sb))
    eigindex = np.argsort(-eigval)
    eigind = eigindex[0: ldafeature]
    rEigvec = eigvec[:, eigind]
    ldadata = data.dot(rEigvec)
    print ldadata.shape
    #plt.plot(range(len(eigindex)),eigval[eigindex])
    #plt.show()
    return ldadata
def ldatest():
    print "lda"
    lda()
if __name__ == "__main__":
    ldatest()
