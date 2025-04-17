

import numpy
import scipy.special
import sklearn.datasets
import matplotlib.pyplot as plt
import os

# Ensure the save directory exists
save_dir = 'save-lr'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

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



# Optimize the logistic regression loss
def trainLogRegBinary(DTR, LTR, l):

    ZTR = LTR * 2.0 - 1.0 # We do it outside the objective function, since we only need to do it once

    def logreg_obj_with_grad(v): # We compute both the objective and its gradient to speed up the optimization
        w = v[:-1]
        b = v[-1]
        s = numpy.dot(vcol(w).T, DTR).ravel() + b

        loss = numpy.logaddexp(0, -ZTR * s)

        G = -ZTR / (1.0 + numpy.exp(ZTR * s))
        GW = (vrow(G) * DTR).mean(1) + l * w.ravel()
        Gb = G.mean()
        return loss.mean() + l / 2 * numpy.linalg.norm(w)**2, numpy.hstack([GW, numpy.array(Gb)])

    vf = scipy.optimize.fmin_l_bfgs_b(logreg_obj_with_grad, x0 = numpy.zeros(DTR.shape[0]+1))[0]
    print ("Log-reg - lambda = %e - J*(w, b) = %e" % (l, logreg_obj_with_grad(vf)[0]))
    return vf[:-1], vf[-1]

# Optimize the weighted logistic regression loss
def trainWeightedLogRegBinary(DTR, LTR, l, pT):

    ZTR = LTR * 2.0 - 1.0 # We do it outside the objective function, since we only need to do it once
    
    wTrue = pT / (ZTR>0).sum() # Compute the weights for the two classes
    wFalse = (1-pT) / (ZTR<0).sum()

    def logreg_obj_with_grad(v): # We compute both the objective and its gradient to speed up the optimization
        w = v[:-1]
        b = v[-1]
        s = numpy.dot(vcol(w).T, DTR).ravel() + b

        loss = numpy.logaddexp(0, -ZTR * s)
        loss[ZTR>0] *= wTrue # Apply the weights to the loss computations
        loss[ZTR<0] *= wFalse

        G = -ZTR / (1.0 + numpy.exp(ZTR * s))
        G[ZTR > 0] *= wTrue # Apply the weights to the gradient computations
        G[ZTR < 0] *= wFalse
        
        GW = (vrow(G) * DTR).sum(1) + l * w.ravel()
        Gb = G.sum()
        return loss.sum() + l / 2 * numpy.linalg.norm(w)**2, numpy.hstack([GW, numpy.array(Gb)])

    vf = scipy.optimize.fmin_l_bfgs_b(logreg_obj_with_grad, x0 = numpy.zeros(DTR.shape[0]+1))[0]
    print ("Weighted Log-reg (pT %e) - lambda = %e - J*(w, b) = %e" % (pT, l, logreg_obj_with_grad(vf)[0]))
    return vf[:-1], vf[-1]


def expand_features_quadratic(D):
    n = D.shape[1]
    expanded_features = [D]
    
    for i in range(D.shape[0]):
        for j in range(i, D.shape[0]):
            expanded_features.append(vrow(D[i, :] * D[j, :]))
    
    return numpy.vstack(expanded_features)



import MVGComparison as bayesRisk # Laboratory 7
import Ploting


if __name__ == '__main__':
    
    D, L = Ploting.load('traindata.txt')
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    
    
    # # Keep only 1 out of 50 samples for training - part 2- uncomment 
    # DTR = DTR[:, ::50]
    # LTR = LTR[::50]
    
    
    # # Expand features to quadratic terms - part 4 - uncomment
    # DTR = expand_features_quadratic(DTR)
    # DVAL = expand_features_quadratic(DVAL)
    
    
    # Center the data - part 5 - uncomment
    # mean_DTR = DTR.mean(axis=1, keepdims=True)
    # DTR = DTR - mean_DTR
    # DVAL = DVAL - mean_DTR
    
    
    
    lambdas = numpy.logspace(-4, 2, 13)
    minDCFs = []
    actDCFs = []
    minDCFs_Weighted = []
    actDCFs_Weighted = []
    for lamb in lambdas:
        w, b = trainLogRegBinary(DTR, LTR, lamb) # Train model
        
        
        sVal = numpy.dot(w.T, DVAL) + b # Compute validation scores
        PVAL = (sVal > 0) * 1 # Predict validation labels - sVal > 0 returns a boolean array, multiplying by 1 (integer) we get an integer array with 0's and 1's corresponding to the original True and False values
        err = (PVAL != LVAL).sum() / float(LVAL.size)
        print ('Error rate: %.1f' % (err*100))
        
        # Save model parameters
        numpy.savez(os.path.join(save_dir, f'model_lambda_{lamb}.npz'), w=w, b=b)
        # Save validation scores and labels
        numpy.savez(os.path.join(save_dir, f'scores_lambda_{lamb}.npz'), sVal=sVal, LVAL=LVAL)
        
        
        
        # Compute empirical prior
        pEmp = (LTR == 1).sum() / LTR.size
        
        # Compute LLR-like scores
        sValLLR = sVal - numpy.log(pEmp / (1-pEmp))
        
        # Compute optimal decisions for the three priors 0.1
        pT = 0.1
        minDCF =  bayesRisk.compute_minDCF_binary_fast(sValLLR, LVAL, pT, 1.0, 1.0)
        actDCF = bayesRisk.compute_actDCF_binary_fast(sValLLR, LVAL, pT, 1.0, 1.0)
        
        minDCFs.append(minDCF)
        actDCFs.append(actDCF)
        
        print('minDCF - pT = 0.1: %.4f' % minDCF)
        print('actDCF - pT = 0.1: %.4f' % actDCF)
        print()
        
        # pT = 0.1
        # w, b = trainWeightedLogRegBinary(DTR, LTR, lamb, pT = pT) # Train model to print the loss
        # sVal = numpy.dot(w.T, DVAL) + b
        # sValLLR = sVal - numpy.log(pT / (1-pT))
        
        
        
        # # Save model parameters
        # numpy.savez(os.path.join(save_dir, f'model_lambda_prior{lamb}.npz'), w=w, b=b)
        # # Save validation scores and labels
        # numpy.savez(os.path.join(save_dir, f'scores_lambda_prior{lamb}.npz'), sVal=sVal, LVAL=LVAL)
        
        
        # minDCF_Weighted = bayesRisk.compute_minDCF_binary_fast(sValLLR, LVAL, pT, 1.0, 1.0)
        # actDCF_Weighted = bayesRisk.compute_actDCF_binary_fast(sValLLR, LVAL, pT, 1.0, 1.0)
        
        
        # minDCFs_Weighted.append(minDCF_Weighted)
        # actDCFs_Weighted.append(actDCF_Weighted)
        # print ('minDCF - pT = 0.1: %.4f' % bayesRisk.compute_minDCF_binary_fast(sValLLR, LVAL, pT, 1.0, 1.0))
        # print ('actDCF - pT = 0.1: %.4f' % bayesRisk.compute_actDCF_binary_fast(sValLLR, LVAL, pT, 1.0, 1.0))
        
        # print ()


    # Plot the DCF metrics as a function of lambda
    # plt.figure()
    # plt.plot(lambdas, minDCFs, label='minDCF - pT=0.1')
    # plt.plot(lambdas, actDCFs, label='actDCF - pT=0.1')
    # plt.xscale('log', base = 10)
    # plt.xlabel('lambda')
    # plt.ylabel('DCF')
    # plt.title('DCF vs Lambda')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    
    
    # # Plot the DCF metrics as a function of lambda
    # plt.figure()
    # plt.plot(lambdas, minDCFs_Weighted, label='minDCF - pT=0.1')
    # plt.plot(lambdas, actDCFs_Weighted, label='actDCF - pT=0.1')
    # plt.xscale('log', base = 10)
    # plt.xlabel('lambda')
    # plt.ylabel('DCF')
    # plt.title('DCF vs Lambda')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    
    
    
    
    