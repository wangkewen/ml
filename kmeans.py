import numpy as np
import matplotlib.pyplot as plt
def load(filename):
    data = []
    fr = open(filename)
    for line in fr.readlines():
        linearr = line.strip().split("\t")
        floatline = map(float,linearr)
        data.append(floatline)
    return np.array(data)
def initCenter(data, k):
    # generate k centers
    # attributes# n
    n = data.shape[1]
    centers = np.zeros((k,n))
    for j in range(n):
        minv = np.min(data[:,j])
        rang = float(np.max(data[:,j]) - minv)
        onecenter = np.random.rand(k,1) * rang + minv
        centers[:, j] = onecenter[:,0]
    return centers
def initCen(data, k):
    centers = []
    m = data.shape[0]
    indexSet = set([])
    i=0
    while i<k:
        index = np.random.randint(0,m)
        if index in indexSet:
            continue
        centers.append(1.0*data[index,:])
        indexSet.add(index)
        i=i+1
    centers=np.array(centers)
    return centers
def km():
    filename = "kmdata/testSet2.txt"
    data = load(filename)
    data = np.array([[1,2],[3,4],[23,23],[24,25],[35,34],[29,30],[77,78],[100,101],[109,97]])
    k = 3
    centers = initCen(data,k)
    m = data.shape[0]
    # node cluster No. and distance
    nodeCluster = np.zeros((m,2))
    itern = 100
    # iterations
    isChange = True
    while isChange:
        isChange = False
        for i in range(m):
            # minimum distance and index
            minDis = np.inf
            minIndex = -1
            for j in range(k):
                # euclidean distance
                disij = np.sqrt(np.sum((data[i,:]-centers[j,:])**2))
                if disij < minDis:
                    minDis = disij
                    minIndex = j
            if nodeCluster[i,0] != minIndex:
                #result is changed
                isChange = True
                nodeCluster[i,:]= minIndex, minDis
        # update new centers
        for j in range(k):
            samecluster = data[np.nonzero(nodeCluster[:,0]==j)]
            #if(len(samecluster)>0):
            centers[j,:] = np.mean(samecluster,axis=0)
    #print(centers.shape)
    #plt.plot(data[:,0], data[:,1], "o")
    #plt.plot(centers[:,0], centers[:,1], "x")
    #plt.show()
    return centers, nodeCluster
if __name__ == "__main__":
    centers, nodeCluster = km()
    print(centers)
