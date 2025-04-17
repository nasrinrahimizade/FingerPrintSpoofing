

import numpy
import sklearn.datasets 
import Ploting

import matplotlib
import matplotlib.pyplot as plt

import scipy.linalg

def load_iris(): # Same as in pca script
    
    import sklearn.datasets
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

def vcol(x): # Same as in pca script
    return x.reshape((x.size, 1))

def vrow(x): # Same as in pca script
    return x.reshape((1, x.size))

def compute_mu_C(D): # Same as in pca script
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

def compute_Sb_Sw(D, L):
    Sb = 0
    Sw = 0
    muGlobal = vcol(D.mean(1))
    for i in numpy.unique(L):
        DCls = D[:, L == i]
        mu = vcol(DCls.mean(1))
        Sb += (mu - muGlobal) @ (mu - muGlobal).T * DCls.shape[1]
        Sw += (DCls - mu) @ (DCls - mu).T
    return Sb / D.shape[1], Sw / D.shape[1]

def compute_lda_geig(D, L, m):
    
    Sb, Sw = compute_Sb_Sw(D, L)
    s, U = scipy.linalg.eigh(Sb, Sw)
    return U[:, ::-1][:, 0:m]

def compute_lda_JointDiag(D, L, m):

    Sb, Sw = compute_Sb_Sw(D, L)

    U, s, _ = numpy.linalg.svd(Sw)
    P = numpy.dot(U * vrow(1.0/(s**0.5)), U.T)

    Sb2 = numpy.dot(P, numpy.dot(Sb, P.T))
    U2, s2, _ = numpy.linalg.svd(Sb2)

    P2 = U2[:, 0:m]
    return numpy.dot(P2.T, P).T

def apply_lda(U, D):
    return U.T @ D

def plot_hist(D, L):

    D0 = D[:, L==0]
    D1 = D[:, L==1]
    

    hFea ={
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

    D, L = Ploting.load('traindata.txt')
    U = compute_lda_geig(D, L, m = 2)
    #W = U
    
    # UW, _, _ = numpy.linalg.svd(W)
    # U = UW[:, 0:m]
    
    print(U)
    print(compute_lda_JointDiag(D, L, m=1)) # May have different signs for the different directions
    W = -1 * compute_lda_JointDiag(D, L, m=1)
    y = numpy.dot(W.T, D) 
    
    plot_hist(y, L)
    
    
    