
from gmm import *
import matplotlib.pyplot as plt
import numpy
import MVGComparison as bayesRisk
import Ploting
from SVM import *
from LogisticRegression import *

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

def plot_dcf_vs_components(components, min_dcf, act_dcf, cov_type):
    """
    Plot minDCF and actDCF against the number of components for a given covariance type.
    """
    plt.plot(components, min_dcf, marker='o', label=f'{cov_type} - minDCF')
    plt.plot(components, act_dcf, marker='x', linestyle='--', label=f'{cov_type} - actDCF')
    plt.xlabel('Number of Components')
    plt.ylabel('DCF')
    plt.title(f'DCF Performance vs. Number of Components for {cov_type.capitalize()} Models')
    
    plt.legend()
    plt.grid(True)

def bayes_error_plot(min_dcf, act_dcf, eff_prior_log_odds, model_name):
    """
    Generate a Bayes error plot for a given model.
    """
    plt.plot(eff_prior_log_odds, min_dcf, label=f'{model_name} - minDCF')
    plt.plot(eff_prior_log_odds, act_dcf, linestyle='--', label=f'{model_name} - actDCF')
    plt.xlabel('Effective Prior Log Odds')
    plt.ylabel('DCF')
    plt.title(f'Bayes Error Plot for {model_name}')
    
    plt.legend()
    plt.grid(True)


if __name__ == '__main__':

    D, L = Ploting.load('traindata.txt')
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    
    

    components = [1, 2, 4, 8, 16, 32]
    min_dcf_full = []
    act_dcf_full = []
    min_dcf_dia = []
    act_dcf_dia = []

    for covType in ['full', 'diagonal']:
        print(covType)
        for numC in components:
            gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], numC, covType=covType, verbose=False, psiEig=0.01)
            gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], numC, covType=covType, verbose=False, psiEig=0.01)

            SLLR = logpdf_GMM(DVAL, gmm1) - logpdf_GMM(DVAL, gmm0)
            min_dcf = bayesRisk.compute_minDCF_binary_fast(SLLR, LVAL, 0.5, 1.0, 1.0)
            act_dcf = bayesRisk.compute_actDCF_binary_fast(SLLR, LVAL, 0.5, 1.0, 1.0)

            if covType == 'full':
                min_dcf_full.append(min_dcf)
                act_dcf_full.append(act_dcf)
            elif covType == 'diagonal':
                min_dcf_dia.append(min_dcf)
                act_dcf_dia.append(act_dcf)

            print(f'numC = {numC}: minDCF = {min_dcf:.4f} / actDCF = {act_dcf:.4f}')

        print()

    # Plotting results for Full Covariance Models
    plt.figure(figsize=(12, 6))
    plot_dcf_vs_components(components, min_dcf_full, act_dcf_full, 'Full')
    plt.show()

    # Plotting results for Tied Covariance Models
    plt.figure(figsize=(12, 6))
    plot_dcf_vs_components(components, min_dcf_dia, act_dcf_dia, 'diagonal')
    plt.show()
    
    
    
    
    # ###############################################
    # #GMM
    
    # components = [ 8]
    # min_dcf_full = []
    # act_dcf_full = []
    # min_dcf_tied = []
    # act_dcf_tied = []

    # for covType in ['tied']:
    #     print(covType)
    #     for numC in components:
    #         gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], numC, covType=covType, verbose=False, psiEig=0.01)
    #         gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], numC, covType=covType, verbose=False, psiEig=0.01)

    #         SLLR = logpdf_GMM(DVAL, gmm1) - logpdf_GMM(DVAL, gmm0)
    #         min_dcf = bayesRisk.compute_minDCF_binary_fast(SLLR, LVAL, 0.5, 1.0, 1.0)
    #         act_dcf = bayesRisk.compute_actDCF_binary_fast(SLLR, LVAL, 0.5, 1.0, 1.0)

    #         if covType == 'full':
    #             min_dcf_full.append(min_dcf)
    #             act_dcf_full.append(act_dcf)
    #         elif covType == 'tied':
    #             min_dcf_tied.append(min_dcf)
    #             act_dcf_tied.append(act_dcf)

    #         print(f'numC = {numC}: minDCF = {min_dcf:.4f} / actDCF = {act_dcf:.4f}')

    #     print()
    
    #bayes error plot
    
    numC = 32 #number of components
    eff_prior_log_odds = numpy.linspace(-4, 4, 100)  # Effective prior log odds range
    eff_priors = 1 / (1 + numpy.exp(-eff_prior_log_odds))  # Effective priors
    
    for covType in ['diagonal']:
        print(covType)
        min_dcf = []
        act_dcf = []
        
        gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], numC, covType=covType, verbose=False, psiEig=0.01)
        gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], numC, covType=covType, verbose=False, psiEig=0.01)
        
        SLLR = logpdf_GMM(DVAL, gmm1) - logpdf_GMM(DVAL, gmm0)
        
        for pi in eff_priors:
            min_dcf_val = bayesRisk.compute_minDCF_binary_fast(SLLR, LVAL, pi, 1.0, 1.0)
            act_dcf_val = bayesRisk.compute_actDCF_binary_fast(SLLR, LVAL, pi, 1.0, 1.0)
            min_dcf.append(min_dcf_val)
            act_dcf.append(act_dcf_val)

        plt.figure(figsize=(12, 6))
        bayes_error_plot(min_dcf, act_dcf, eff_prior_log_odds, f'GMM ({covType.capitalize()})')
        plt.show()
    
    # ##############
    # #SVM
    # gammas = [ 10**-1]
    # Cs = [1]
    # for gamma in gammas:
    #     kernelFunc = rbfKernel(gamma)
    #     minDCFs_gamma = []
    #     actDCFs_gamma = []
    #     for C in Cs:
    #         fScore = train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps=1.0)
    #         SVAL = fScore(DVAL)
    #         np.save(os.path.join(save_dir, f'SVAL_rbf_gamma_{gamma}_C_{C}.npy'), SVAL)
    #         np.save(os.path.join(save_dir, f'LVAL_rbf_gamma_{gamma}_C_{C}.npy'), LVAL)
    #         PVAL = (SVAL > 0) * 1
    #         err = (PVAL != LVAL).sum() / float(LVAL.size)
    #         print(f'Error rate: {err*100:.1f}%')
    #         minDCF = bayesRisk.compute_minDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
    #         actDCF = bayesRisk.compute_actDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
    #         minDCFs_gamma.append(minDCF)
    #         actDCFs_gamma.append(actDCF)
    #         print(f'minDCF - pT = 0.1: {minDCF:.4f}')
    #         print(f'actDCF - pT = 0.1: {actDCF:.4f}')
    #         print()
    #     minDCFs.append(minDCFs_gamma)
    #     actDCFs.append(actDCFs_gamma)
    
    # plot_metrics2(Cs, minDCFs, actDCFs, gammas, title_suffix=' (pT=0.1)')
    
    #bayes error plot
    # SVM with RBF Kernel Analysis
    C = 1
    gamma = 0.1

    # Train the SVM model with RBF kernel
    kernelFunc = rbfKernel(gamma)
    fScore = train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps=1.0)

    # Compute validation scores
    SVAL = fScore(DVAL)

    # Compute DCFs for a range of effective priors
    min_dcf_svm = []
    act_dcf_svm = []

    for pi in eff_priors:
        min_dcf_val = bayesRisk.compute_minDCF_binary_fast(SVAL, LVAL, pi, 1.0, 1.0)
        act_dcf_val = bayesRisk.compute_actDCF_binary_fast(SVAL, LVAL, pi, 1.0, 1.0)
        min_dcf_svm.append(min_dcf_val)
        act_dcf_svm.append(act_dcf_val)

    plt.figure(figsize=(12, 6))
    bayes_error_plot(min_dcf_svm, act_dcf_svm, eff_prior_log_odds, 'SVM with RBF Kernel')
    plt.show()    
    
    
    # ####################
    # #LR
    
    
    # DTR = expand_features_quadratic(DTR)
    # DVAL = expand_features_quadratic(DVAL)
    
    
    # lambdas = numpy.logspace(-4, 2, 13)
    # minDCFs = []
    # actDCFs = []
    # minDCFs_Weighted = []
    # actDCFs_Weighted = []
    # for lamb in lambdas:
    #     w, b = trainLogRegBinary(DTR, LTR, lamb) # Train model
        
        
    #     sVal = numpy.dot(w.T, DVAL) + b # Compute validation scores
    #     PVAL = (sVal > 0) * 1 # Predict validation labels - sVal > 0 returns a boolean array, multiplying by 1 (integer) we get an integer array with 0's and 1's corresponding to the original True and False values
    #     err = (PVAL != LVAL).sum() / float(LVAL.size)
    #     print ('Error rate: %.1f' % (err*100))
        
    #     # Save model parameters
    #     numpy.savez(os.path.join(save_dir, f'model_lambda_{lamb}.npz'), w=w, b=b)
    #     # Save validation scores and labels
    #     numpy.savez(os.path.join(save_dir, f'scores_lambda_{lamb}.npz'), sVal=sVal, LVAL=LVAL)
        
        
        
    #     # Compute empirical prior
    #     pEmp = (LTR == 1).sum() / LTR.size
        
    #     # Compute LLR-like scores
    #     sValLLR = sVal - numpy.log(pEmp / (1-pEmp))
        
    #     # Compute optimal decisions for the three priors 0.1
    #     pT = 0.1
    #     minDCF =  bayesRisk.compute_minDCF_binary_fast(sValLLR, LVAL, pT, 1.0, 1.0)
    #     actDCF = bayesRisk.compute_actDCF_binary_fast(sValLLR, LVAL, pT, 1.0, 1.0)
        
    #     minDCFs.append(minDCF)
    #     actDCFs.append(actDCF)
        
    #     print('minDCF - pT = 0.1: %.4f' % minDCF)
    #     print('actDCF - pT = 0.1: %.4f' % actDCF)
    #     print()
        
    
    #bayes error plot
    
    # Quadratic Logistic Regression Analysis
    lambda_value = 3.162 * 10e-2
    
    # Expand features for quadratic logistic regression
    DTR_expanded = expand_features_quadratic(DTR)
    DVAL_expanded = expand_features_quadratic(DVAL)

    # Train the logistic regression model
    w, b = trainLogRegBinary(DTR_expanded, LTR, lambda_value)

    # Compute validation scores
    sVal = numpy.dot(w.T, DVAL_expanded) + b

    # Compute DCFs for a range of effective priors
    min_dcf_lr = []
    act_dcf_lr = []

    for pi in eff_priors:
        min_dcf_val = bayesRisk.compute_minDCF_binary_fast(sVal, LVAL, pi, 1.0, 1.0)
        act_dcf_val = bayesRisk.compute_actDCF_binary_fast(sVal, LVAL, pi, 1.0, 1.0)
        min_dcf_lr.append(min_dcf_val)
        act_dcf_lr.append(act_dcf_val)

    plt.figure(figsize=(12, 6))
    bayes_error_plot(min_dcf_lr, act_dcf_lr, eff_prior_log_odds, 'Quadratic Logistic Regression')
    plt.show()