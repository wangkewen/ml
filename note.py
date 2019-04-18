import numpy as np
class Emsemble:
    def __init__(self):
	print("pseudo code")
    def decisionTree(x, y, features):
        print("create decision tree")
        # x: independent variables
        # y: labels
        return tree
    
    #https://towardsdatascience.com/random-forests-and-decision-trees-from-scratch-in-python-3e4fa5ae4249
    def randomForest(x, y):
	# number of decision trees of RF
	n = 100
        # train this tree
	trees = []
	for i in range(n):
        # can be parallel
            ni = x.shape[0]
            index = np.random.permutation(x.shape[0])[:ni]
            rx, ry = x[index], y[index]
            nf = int(np.sqrt(x.shape[1]))       
            rfeatures = np.random.permutation(nf)
	    trees.append(decisionTree(rx, ry, rfeatures))
        
        predicts = np.mean([tree.predict(x) for tree in trees])
    #https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d
    #https://medium.com/@cwchang/gradient-boosting-%E7%B0%A1%E4%BB%8B-f3a578ae7205
    def gradientBoostingTree(x, y):
        # number of decision trees of GBT
        n = 30
        ypred = 0
        yi = y
        # y ~ F(x)
        # loss function
        # L(y, F(x)) = (1/2)*(y-F(x))^2
        # Gradient = y-F(x)
        # Residual h(x) = y-F(x)
        for i in range(n):
            tree = decisionTree(x, yi, [j for j in range(x.shape[1])])
            ypredi = tree.predict(x)
            ypred = ypredi + ypred
            ei = y - ypred
            y = ei
