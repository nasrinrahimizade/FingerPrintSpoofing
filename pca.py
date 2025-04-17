
import numpy
import sklearn.datasets 
import Ploting

import matplotlib
import matplotlib.pyplot as plt

import scipy.linalg

def load_iris():

    import sklearn.datasets
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

def compute_pca(D, m):

    mu, C = compute_mu_C(D)
    U, s, Vh = numpy.linalg.svd(C)
    P = U[:, 0:m]
    return P

def apply_pca(P, D):
    return P.T @ D
    

   


def plot_scatter(D, L):
    
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    

    hFea = {
        0: 'f1',
        1: 'f2',
        2: 'f3',
        3: 'f4',
        4: 'f5',
        5: 'f6'
        }

    for dIdx1 in range (5):
        for dIdx2 in range (5):
            if dIdx1 == dIdx2:
                continue
            plt.figure()
            plt.xlabel(hFea[dIdx1])
            plt.ylabel(hFea[dIdx2])
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label = 'genuine-true')
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label = 'fake-false')
            
        
            plt.legend()
            plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
            plt.savefig('scatter_%d_%d.jpg' % (dIdx1, dIdx2))
        plt.show()
        
def plot_hist(D, L):

    D0 = D[:, L==0]
    D1 = D[:, L==1]
    

    hFea = {
        0: 'f1',
        1: 'f2',
        2: 'f3',
        3: 'f4',
        4: 'f5',
        5: 'f6'
        }

    for dIdx in range(6):
        plt.figure()
        plt.xlabel(hFea[dIdx])
        plt.hist(D0[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'genuine-true')
        plt.hist(D1[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'fake-false')
        
        
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        plt.savefig('hist_%d.jpg' % dIdx)
    plt.show()

        
if __name__ == '__main__':
    
    # Change default font size - comment to use default values
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    D, L = Ploting.load('traindata.txt')
    
    
    mu, C = compute_mu_C(D)
    print(mu)
    print(C)
    P = compute_pca(D, m = 6)
    print(P)
    
    DP = numpy.dot(P.T, D)
    
    plot_hist(DP, L)