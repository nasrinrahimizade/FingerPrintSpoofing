

import numpy
import scipy.special
import sklearn.datasets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np  # For saving data
import os

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

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


# Optimize SVM
def train_dual_SVM_linear(DTR, LTR, C, K = 1):
    
    ZTR = LTR * 2.0 - 1.0 # Convert labels to +1/-1
    DTR_EXT = numpy.vstack([DTR, numpy.ones((1,DTR.shape[1])) * K])
    H = numpy.dot(DTR_EXT.T, DTR_EXT) * vcol(ZTR) * vrow(ZTR)

    # Dual objective with gradient
    def fOpt(alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - numpy.ones(alpha.size)
        return loss, grad

    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(fOpt, numpy.zeros(DTR_EXT.shape[1]), bounds = [(0, C) for i in LTR], factr=1.0)
    
    # Primal loss
    def primalLoss(w_hat):
        S = (vrow(w_hat) @ DTR_EXT).ravel()
        return 0.5 * numpy.linalg.norm(w_hat)**2 + C * numpy.maximum(0, 1 - ZTR * S).sum()

    # Compute primal solution for extended data matrix
    w_hat = (vrow(alphaStar) * vrow(ZTR) * DTR_EXT).sum(1)
    
    # Extract w and b - alternatively, we could construct the extended matrix for the samples to score and use directly v
    w, b = w_hat[0:DTR.shape[0]], w_hat[-1] * K # b must be rescaled in case K != 1, since we want to compute w'x + b * K

    primalLoss, dualLoss = primalLoss(w_hat), -fOpt(alphaStar)[0]
    print ('SVM - C %e - K %e - primal loss %e - dual loss %e - duality gap %e' % (C, K, primalLoss, dualLoss, primalLoss - dualLoss))
    
    return w, b

# We create the kernel function. Since the kernel function may need additional parameters, we create a function that creates on the fly the required kernel function
# The inner function will be able to access the arguments of the outer function
def polyKernel(degree, c):
    
    def polyKernelFunc(D1, D2):
        return (numpy.dot(D1.T, D2) + c) ** degree

    return polyKernelFunc

def rbfKernel(gamma):

    def rbfKernelFunc(D1, D2):
        # Fast method to compute all pair-wise distances. Exploit the fact that |x-y|^2 = |x|^2 + |y|^2 - 2 x^T y, combined with broadcasting
        D1Norms = (D1**2).sum(0)
        D2Norms = (D2**2).sum(0)
        Z = vcol(D1Norms) + vrow(D2Norms) - 2 * numpy.dot(D1.T, D2)
        return numpy.exp(-gamma * Z)

    return rbfKernelFunc

# kernelFunc: function that computes the kernel matrix from two data matrices
def train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps = 1.0):

    ZTR = LTR * 2.0 - 1.0 # Convert labels to +1/-1
    K = kernelFunc(DTR, DTR) + eps
    H = vcol(ZTR) * vrow(ZTR) * K

    # Dual objective with gradient
    def fOpt(alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - numpy.ones(alpha.size)
        return loss, grad

    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(fOpt, numpy.zeros(DTR.shape[1]), bounds = [(0, C) for i in LTR], factr=1.0)

    print ('SVM (kernel) - C %e - dual loss %e' % (C, -fOpt(alphaStar)[0]))

    # Function to compute the scores for samples in DTE
    def fScore(DTE):
        
        K = kernelFunc(DTR, DTE) + eps
        H = vcol(alphaStar) * vcol(ZTR) * K
        return H.sum(0)

    return fScore # we directly return the function to score a matrix of test samples

def plot_metrics2(Cs, minDCFs, actDCFs, gammas, title_suffix=''):
    plt.figure()
    for i, gamma in enumerate(gammas):
        plt.plot(Cs, minDCFs[i], label=f'minDCF - gamma={gamma}')
        plt.plot(Cs, actDCFs[i], label=f'actDCF - gamma={gamma}')
    plt.xscale('log', base=10)
    plt.xlabel('C')
    plt.ylabel('DCF')
    plt.title(f'DCF vs C (RBF Kernel{title_suffix})')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_metrics(Cs, minDCFs, actDCFs):
    plt.figure()
    plt.plot(Cs, minDCFs, label='minDCF')
    plt.plot(Cs, actDCFs, label='actDCF')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('DCF')
    plt.title('DCF vs C (polynominal kernel SVM)')
    plt.legend()
    plt.grid(True)
    plt.show()    
    
def center_data(D):
    """Center the data by subtracting the mean of each feature."""
    mean = D.mean(axis=1, keepdims=True)
    D_centered = D - mean
    return D_centered

    
import MVGComparison as bayesRisk
import Ploting

if __name__ == '__main__':

    D, L = Ploting.load('traindata.txt')
    # Center the data
    # D = center_data(D)
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    
    # Create the save directory if it doesn't exist
    save_dir = "save-svm-scores"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
        
    Cs = numpy.logspace(-5, 0, 11)
    minDCFs = []
    actDCFs = []
    
    # for K in [1]:
    #     for C in Cs:
    #         w, b = train_dual_SVM_linear(DTR, LTR, C, K)
    #         SVAL = (vrow(w) @ DVAL + b).ravel()
    #         PVAL = (SVAL > 0) * 1
    #         err = (PVAL != LVAL).sum() / float(LVAL.size)
    #         print ('Error rate: %.1f' % (err*100))
    #         minDCF = bayesRisk.compute_minDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
    #         actDCF = bayesRisk.compute_actDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
    #         minDCFs.append(minDCF)
    #         actDCFs.append(actDCF)
    #         np.save(os.path.join(save_dir, f'SVAL_linear_C_{C}.npy'), SVAL)
    #         np.save(os.path.join(save_dir, f'LVAL_linear_C_{C}.npy'), LVAL)
    #         print('minDCF - pT = 0.1: %.4f' % minDCF)
    #         print('actDCF - pT = 0.1: %.4f' % actDCF)
    #         print ()

    # plot_metrics(Cs, minDCFs, actDCFs)
    
    
    # print('polynomial kernel')
    # kernelFunc = polyKernel(2, 1)
    
    # for C in Cs:
    #     fScore = train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps=0.0)
    #     SVAL = fScore(DVAL)
    #     np.save(os.path.join(save_dir, f'SVAL_poly_C_{C}.npy'), SVAL)
    #     np.save(os.path.join(save_dir, f'LVAL_poly_C_{C}.npy'), LVAL)
    #     PVAL = (SVAL > 0) * 1
    #     err = (PVAL != LVAL).sum() / float(LVAL.size)
    #     print(f'Error rate: {err*100:.1f}%')
    #     minDCF = bayesRisk.compute_minDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
    #     actDCF = bayesRisk.compute_actDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
    #     minDCFs.append(minDCF)
    #     actDCFs.append(actDCF)
    #     print(f'minDCF - pT = 0.1: {minDCF:.4f}')
    #     print(f'actDCF - pT = 0.1: {actDCF:.4f}')
    #     print()
    
    # plot_metrics(Cs, minDCFs, actDCFs)
        
            
    #RBF kernel
    
    gammas = [10**-4, 10**-3, 10**-2, 10**-1]
    
    for gamma in gammas:
        kernelFunc = rbfKernel(gamma)
        minDCFs_gamma = []
        actDCFs_gamma = []
        for C in Cs:
            fScore = train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps=1.0)
            SVAL = fScore(DVAL)
            np.save(os.path.join(save_dir, f'SVAL_rbf_gamma_{gamma}_C_{C}.npy'), SVAL)
            np.save(os.path.join(save_dir, f'LVAL_rbf_gamma_{gamma}_C_{C}.npy'), LVAL)
            PVAL = (SVAL > 0) * 1
            err = (PVAL != LVAL).sum() / float(LVAL.size)
            print(f'Error rate: {err*100:.1f}%')
            minDCF = bayesRisk.compute_minDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
            actDCF = bayesRisk.compute_actDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
            minDCFs_gamma.append(minDCF)
            actDCFs_gamma.append(actDCF)
            print(f'minDCF - pT = 0.1: {minDCF:.4f}')
            print(f'actDCF - pT = 0.1: {actDCF:.4f}')
            print()
        minDCFs.append(minDCFs_gamma)
        actDCFs.append(actDCFs_gamma)
    
    plot_metrics2(Cs, minDCFs, actDCFs, gammas, title_suffix=' (pT=0.1)')