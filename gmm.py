############################################################################################
# Copyright (C) 2024 by Sandro Cumani                                                      #
#                                                                                          #
# This file is provided for didactic purposes only, according to the Politecnico di Torino #
# policies on didactic material.                                                           #
#                                                                                          #
# Any form of re-distribution or online publication is forbidden.                          #
#                                                                                          #
# This file is provided as-is, without any warranty                                        #
############################################################################################

import numpy
import scipy
import scipy.special
import matplotlib.pyplot as plt

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def logpdf_GAU_ND(x, mu, C): # Fast version from Lab 4
    P = numpy.linalg.inv(C)
    return -0.5*x.shape[0]*numpy.log(numpy.pi*2) - 0.5*numpy.linalg.slogdet(C)[1] - 0.5 * ((x-mu) * (P @ (x-mu))).sum(0)

def split_db_2to1(D, L, seed=0):
    
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    
    return (DTR, LTR), (DVAL, LVAL)

def load_iris():
    
    import sklearn.datasets
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

######
# from GMM_load.py
import json

def save_gmm(gmm, filename):
    gmmJson = [(i, j.tolist(), k.tolist()) for i, j, k in gmm]
    with open(filename, 'w') as f:
        json.dump(gmmJson, f)
    
def load_gmm(filename):
    with open(filename, 'r') as f:
        gmm = json.load(f)
    return [(i, numpy.asarray(j), numpy.asarray(k)) for i, j, k in gmm]
######

def logpdf_GMM(X, gmm):

    S = []
    
    for w, mu, C in gmm:
        logpdf_conditional = logpdf_GAU_ND(X, mu, C)
        logpdf_joint = logpdf_conditional + numpy.log(w)
        S.append(logpdf_joint)
        
    S = numpy.vstack(S)
    logdens = scipy.special.logsumexp(S, axis=0)
    return logdens

def smooth_covariance_matrix(C, psi):

    U, s, Vh = numpy.linalg.svd(C)
    s[s<psi]=psi
    CUpd = U @ (vcol(s) * U.T)
    return CUpd

# X: Data matrix
# gmm: input gmm
# covType: 'Full' | 'Diagonal' | 'Tied'
# psiEig: factor for eignvalue thresholding
#
# return: updated gmm
def train_GMM_EM_Iteration(X, gmm, covType = 'Full', psiEig = None): 

    assert (covType.lower() in ['full', 'diagonal', 'tied'])
    
    # E-step
    S = []
    
    for w, mu, C in gmm:
        logpdf_conditional = logpdf_GAU_ND(X, mu, C)
        logpdf_joint = logpdf_conditional + numpy.log(w)
        S.append(logpdf_joint)
        
    S = numpy.vstack(S) # Compute joint densities f(x_i, c), i=1...n, c=1...G
    logdens = scipy.special.logsumexp(S, axis=0) # Compute marginal for samples f(x_i)

    # Compute posterior for all clusters - log P(C=c|X=x_i) = log f(x_i, c) - log f(x_i)) - i=1...n, c=1...G
    # Each row for gammaAllComponents corresponds to a Gaussian component
    # Each column corresponds to a sample (similar to the matrix of class posterior probabilities in Lab 5, but here the rows are associated to clusters rather than to classes
    gammaAllComponents = numpy.exp(S - logdens)

    # M-step
    gmmUpd = []
    for gIdx in range(len(gmm)): 
    # Compute statistics:
        gamma = gammaAllComponents[gIdx] # Extract the responsibilities for component gIdx
        Z = gamma.sum()
        F = vcol((vrow(gamma) * X).sum(1)) # Exploit broadcasting to compute the sum
        S = (vrow(gamma) * X) @ X.T
        muUpd = F/Z
        CUpd = S/Z - muUpd @ muUpd.T
        wUpd = Z / X.shape[1]
        if covType.lower() == 'diagonal':
            CUpd  = CUpd * numpy.eye(X.shape[0]) # An efficient implementation would store and employ only the diagonal terms, but is out of the scope of this script
        gmmUpd.append((wUpd, muUpd, CUpd))

    if covType.lower() == 'tied':
        CTied = 0
        for w, mu, C in gmmUpd:
            CTied += w * C
        gmmUpd = [(w, mu, CTied) for w, mu, C in gmmUpd]

    if psiEig is not None:
        gmmUpd = [(w, mu, smooth_covariance_matrix(C, psiEig)) for w, mu, C in gmmUpd]
        
    return gmmUpd

# Train a GMM until the average dela log-likelihood becomes <= epsLLAverage
def train_GMM_EM(X, gmm, covType = 'Full', psiEig = None, epsLLAverage = 1e-6, verbose=True):

    llOld = logpdf_GMM(X, gmm).mean()
    llDelta = None
    if verbose:
        print('GMM - it %3d - average ll %.8e' % (0, llOld))
    it = 1
    while (llDelta is None or llDelta > epsLLAverage):
        gmmUpd = train_GMM_EM_Iteration(X, gmm, covType = covType, psiEig = psiEig)
        llUpd = logpdf_GMM(X, gmmUpd).mean()
        llDelta = llUpd - llOld
        if verbose:
            print('GMM - it %3d - average ll %.8e' % (it, llUpd))
        gmm = gmmUpd
        llOld = llUpd
        it = it + 1

    if verbose:
        print('GMM - it %3d - average ll %.8e (eps = %e)' % (it, llUpd, epsLLAverage))        
    return gmm
    
def split_GMM_LBG(gmm, alpha = 0.1, verbose=True):

    gmmOut = []
    if verbose:
        print ('LBG - going from %d to %d components' % (len(gmm), len(gmm)*2))
    for (w, mu, C) in gmm:
        U, s, Vh = numpy.linalg.svd(C)
        d = U[:, 0:1] * s[0]**0.5 * alpha
        gmmOut.append((0.5 * w, mu - d, C))
        gmmOut.append((0.5 * w, mu + d, C))
    return gmmOut

# Train a full model using LBG + EM, starting from a single Gaussian model, until we have numComponents components. lbgAlpha is the value 'alpha' used for LBG, the otehr parameters are the same as in the EM functions above
def train_GMM_LBG_EM(X, numComponents, covType = 'Full', psiEig = None, epsLLAverage = 1e-6, lbgAlpha = 0.1, verbose=True):

    mu, C = compute_mu_C(X)

    if covType.lower() == 'diagonal':
        C = C * numpy.eye(X.shape[0]) # We need an initial diagonal GMM to train a diagonal GMM
    
    if psiEig is not None:
        gmm = [(1.0, mu, smooth_covariance_matrix(C, psiEig))] # 1-component model - if we impose the eignevalus constraint, we must do it for the initial 1-component GMM as well
    else:
        gmm = [(1.0, mu, C)] # 1-component model
    
    while len(gmm) < numComponents:
        # Split the components
        if verbose:
            print ('Average ll before LBG: %.8e' % logpdf_GMM(X, gmm).mean())
        gmm = split_GMM_LBG(gmm, lbgAlpha, verbose=verbose)
        if verbose:
            print ('Average ll after LBG: %.8e' % logpdf_GMM(X, gmm).mean()) # NOTE: just after LBG the ll CAN be lower than before the LBG - LBG does not optimize the ll, it just increases the number of components
        # Run the EM for the new GMM
        gmm = train_GMM_EM(X, gmm, covType = covType, psiEig = psiEig, verbose=verbose, epsLLAverage = epsLLAverage)
    return gmm

    
if __name__ == '__main__':

    X = numpy.load('Data/GMM_data_4D.npy')
    gmm = load_gmm('Data/GMM_4D_3G_init.json')
    llPrecomputed = numpy.load('Data/GMM_4D_3G_init_ll.npy')
    ll = logpdf_GMM(X, gmm)
    print (numpy.abs(ll-llPrecomputed).max()) # Check max absolute difference
    
    X = numpy.load('Data/GMM_data_1D.npy')
    gmm = load_gmm('Data/GMM_1D_3G_init.json')
    llPrecomputed = numpy.load('Data/GMM_1D_3G_init_ll.npy')
    ll = logpdf_GMM(X, gmm)
    print (numpy.abs(ll-llPrecomputed).max()) # Check max absolute difference

    print()
    print('***** EM - 4D *****')
    print()
    X = numpy.load('Data/GMM_data_4D.npy')
    gmm = load_gmm('Data/GMM_4D_3G_init.json')
    gmm = train_GMM_EM(X, gmm)
    print ('Final average ll: %.8e' % logpdf_GMM(X, gmm).mean())

    print()
    print('***** EM - 1D *****')
    print()
    X = numpy.load('Data/GMM_data_1D.npy')
    gmm = load_gmm('Data/GMM_1D_3G_init.json')
    gmm = train_GMM_EM(X, gmm)
    print ('Final average ll: %.8e' % logpdf_GMM(X, gmm).mean())
    
    plt.figure()
    plt.hist(X.ravel(), 25, density=True) # Pay attention to the shape of X: X is a data matrix, so it's a 1 x N array, not a 1-D array
    _X = numpy.linspace(-10, 5, 1000) # Plot gmm density in range (-10, 5) - x-data for the plot
    plt.plot(_X.ravel(), numpy.exp(logpdf_GMM(vrow(_X), gmm))) # Pay attention to the shape of _X: for plotting _X should be a 1-D array, for logpdf_GMM it should be a 1 x N matrix with one-dimensional samples

    print()
    print('***** LBG EM - 4D *****')
    print()
    X = numpy.load('Data/GMM_data_4D.npy')
    gmm = train_GMM_LBG_EM(X, 4)
    print ('LBG + EM - final average ll: %.8e (%d components)' % (logpdf_GMM(X, gmm).mean(), len(gmm)))
    print ('LBG + EM - final average ll - pretrained model: %.8e' % (logpdf_GMM(X, load_gmm('Data/GMM_4D_4G_EM_LBG.json')).mean()))
    #print(gmm) # you can print the gmms
    #print(load_gmm('Data/GMM_4D_4G_EM_LBG.json')) # you can print the gmms
    print ('Max absolute ll difference w.r.t. pre-trained model over all training samples:', (numpy.abs(logpdf_GMM(X, gmm) - logpdf_GMM(X, load_gmm('Data/GMM_4D_4G_EM_LBG.json')))).max())

    print()
    print('***** LBG EM - 1D *****')
    print()
    X = numpy.load('Data/GMM_data_1D.npy')
    gmm = train_GMM_LBG_EM(X, 4)
    print ('LBG + EM - final average ll: %.8e (%d components)' % (logpdf_GMM(X, gmm).mean(), len(gmm)))
    print ('LBG + EM - final average ll - pretrained model: %.8e' % (logpdf_GMM(X, load_gmm('Data/GMM_1D_4G_EM_LBG.json')).mean()))
    #print(gmm) # you can print the gmms
    #print(load_gmm('Data/GMM_1D_4G_EM_LBG.json')) # you can print the gmms
    print ('Max absolute ll difference w.r.t. pre-trained model over all training samples:', (numpy.abs(logpdf_GMM(X, gmm) - logpdf_GMM(X, load_gmm('Data/GMM_1D_4G_EM_LBG.json')))).max())

    plt.figure()
    plt.hist(X.ravel(), 25, density=True) # Pay attention to the shape of X: X is a data matrix, so it's a 1 x N array, not a 1-D array
    _X = numpy.linspace(-10, 5, 1000) # Plot gmm density in range (-10, 5) - x-data for the plot
    plt.plot(_X.ravel(), numpy.exp(logpdf_GMM(vrow(_X), gmm)), 'r') # Pay attention to the shape of _X: for plotting _X should be a 1-D array, for logpdf_GMM it should be a 1 x N matrix with one-dimensional samples
    #plt.show()
    
    print()
    print('***** LBG EM - 4D - Eigenvalue Theshold *****')
    print()
    X = numpy.load('Data/GMM_data_4D.npy')
    psiEig = 0.01  # Set the eigenvalue threshold
    gmm = train_GMM_LBG_EM(X, 4, psiEig=psiEig)

    # gmm = train_GMM_LBG_EM(X, 4)
    print ('LBG + EM - final average ll: %.8e (%d components)' % (logpdf_GMM(X, gmm).mean(), len(gmm)))
    print ('LBG + EM - final average ll - pretrained model: %.8e' % (logpdf_GMM(X, load_gmm('Data/GMM_4D_4G_EM_LBG.json')).mean()))
    #print(gmm) # you can print the gmms
    #print(load_gmm('Data/GMM_4D_4G_EM_LBG.json')) # you can print the gmms
    print ('Max absolute ll difference w.r.t. pre-trained model over all training samples:', (numpy.abs(logpdf_GMM(X, gmm) - logpdf_GMM(X, load_gmm('Data/GMM_4D_4G_EM_LBG.json')))).max())

    