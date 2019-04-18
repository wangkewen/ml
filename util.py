import numpy as np
def test():
    data = np.array([[1,2,3],[2,3,1],[2,3,2],[1,2,5]])
    #print(data[np.nonzero(data[:,0]==1)])
    #print(np.sqrt(np.sum(data**2)))
    #print(data + 100)
    res = np.zeros((2,3))
    dmean = np.mean(data, axis=0)
    #print(dmean)
    #print(dmean+3)
    res[0,:] = dmean
    #print(res)
    freq = [1,1,1,1,1,1,2,2,2,2,2,3,3,3,3]
    fcount = np.bincount(freq)
    fcum = np.cumsum(fcount)
    #print(fcount)
    #print(fcum)
    ranp = np.random.rand(3,1)
    #print(ranp)
    #print(np.inf>1.7e308)
    l = ["aa","bak","aa"]
    m = ["ut","pp","pp","bak"]
    print(list(set(l)|set(m)))
    dda = np.array([[1,2,3],[1,1,1]])
    ddb = np.array([[-1,-1,-1],[0,1,0]])
    print(np.log(100 / dda))
    ddc = dda * ddb
    print(np.random.permutation(10))
    print(np.sqrt([i for i in range(10)]))
if __name__ == "__main__":
    test()
