########################################################
# Serafeim Loukas, Last modified: 10/2017 

#Import the needed modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA
from scipy import stats



def biplot(score,coeff,labels=None):
	#Inputs : Component scores also called factor scores (the transformed variable values corresponding to a particular data point), and loadings (the weight by which each standardized original variable should be multiplied to get the component score)
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley, c = y)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel("PC{}".format(1))
plt.ylabel("PC{}".format(2))
plt.grid()
 
#Use Iris data to test the code

iris = datasets.load_iris()
X = iris.data
y = iris.target

X = stats.zscore(X)
#Apply PCA and then use the biplot function to plot the biplot.
pca = PCA(X) 
biplot(pca.Y[:,0:2],pca.Wt[:,0:2])
plt.show()
