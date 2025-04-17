import numpy

import logReg
import matplotlib
import matplotlib.pyplot as plt
from gmm import *
import MVGComparison as bayesRisk
from Ploting import *
from SVM import *
from LogisticRegression import *

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def bayesPlot(S, L, left = -3, right = 3, npts = 21):
    
    effPriorLogOdds = numpy.linspace(left, right, npts)
    effPriors = 1.0 / (1.0 + numpy.exp(-effPriorLogOdds))
    actDCF = []
    minDCF = []
    for effPrior in effPriors:
        actDCF.append(bayesRisk.compute_actDCF_binary_fast(S, L, effPrior, 1.0, 1.0))
        minDCF.append(bayesRisk.compute_minDCF_binary_fast(S, L, effPrior, 1.0, 1.0))
    return effPriorLogOdds, actDCF, minDCF

# Extract i-th fold from a 1-D numpy array (as for the single fold case, we do not need to shuffle scores in this case, but it may be necessary if samples are sorted in peculiar ways to ensure that validation and calibration sets are independent and with similar characteristics   
def extract_train_val_folds_from_ary(X, idx):
    return numpy.hstack([X[jdx::KFOLD] for jdx in range(KFOLD) if jdx != idx]), X[idx::KFOLD]

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
    
# Feature expansion to quadratic terms
def expand_features_quadratic(D):
    n = D.shape[1]
    expanded_features = [D]
    
    for i in range(D.shape[0]):
        for j in range(i, D.shape[0]):
            expanded_features.append(vrow(D[i, :] * D[j, :]))
    
    return numpy.vstack(expanded_features)

# Preprocessing methods
def preprocess_data(D, method):
    if method == 'center':
        mean_D = D.mean(axis=1, keepdims=True)
        return D - mean_D
    elif method == 'znormalize':
        mean_D = D.mean(axis=1, keepdims=True)
        std_D = D.std(axis=1, ddof=1, keepdims=True)
        return (D - mean_D) / std_D
    elif method == 'whiten':
        mean_D = D.mean(axis=1, keepdims=True)
        std_D = D.std(axis=1, ddof=1, keepdims=True)
        D_centered = (D - mean_D) / std_D
        cov_matrix = np.cov(D_centered)
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        eigvals = np.maximum(eigvals, 1e-5)  # Avoid division by zero
        whitening_matrix = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
        return whitening_matrix @ D_centered
    elif method == 'expand':
        return expand_features_quadratic(D)
    elif method == 'center_expand':
        mean_D = D.mean(axis=1, keepdims=True)
        D_centered = D - mean_D
        return expand_features_quadratic(D_centered)
    else:
        return D

    
if __name__ == '__main__':

    SAMEFIGPLOTS = True # set to False to have 1 figure per plot
    
    
    # Load the evaluation data
    D_eval, L_eval = load('evalData.txt')
    
    
    D, L = load('traindata.txt')
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    
    numC= 32
    covType = 'diagonal'
    gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], numC, covType=covType, verbose=False, psiEig=0.01)
    gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], numC, covType=covType, verbose=False, psiEig=0.01)
    
    SLLR_gmm  = logpdf_GMM(DVAL, gmm1) - logpdf_GMM(DVAL, gmm0)
    eval_scores_GMM  = logpdf_GMM(D_eval, gmm1) - logpdf_GMM(D_eval, gmm0)
    
    
    
    C = 1
    gamma = 0.1

    # Train the SVM model with RBF kernel
    kernelFunc = rbfKernel(gamma)
    fScore = train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps=1.0)

    # Compute validation scores
    SVAL_svm = fScore(DVAL)
    eval_scores_SVM = fScore(D_eval)
    
    
    
    
    # Quadratic Logistic Regression Analysis
    lambda_value = 3.162 * 10e-2
    
    # Expand features for quadratic logistic regression
    DTR_expanded = expand_features_quadratic(DTR)
    DVAL_expanded = expand_features_quadratic(DVAL)
    DEVAL_expanded = expand_features_quadratic(D_eval)
    
    # Train the logistic regression model
    w, b = trainLogRegBinary(DTR_expanded, LTR, lambda_value)

    # Compute validation scores
    SVAL_lr = numpy.dot(w.T, DVAL_expanded) + b
    eval_scores_LR = numpy.dot(w.T, DEVAL_expanded) + b
    
    
    
    if SAMEFIGPLOTS:
        fig = plt.figure(figsize=(20,20))
        axes = fig.subplots(4,3, sharex='all')
        fig.suptitle('K-fold')
    else:
        axes = numpy.array([ [plt.figure().gca(), plt.figure().gca(), plt.figure().gca()], [plt.figure().gca(), plt.figure().gca(), plt.figure().gca()], [None, plt.figure().gca(), plt.figure().gca()] ])

    print()
    print('*** K-FOLD ***')
    print()
    
    KFOLD = 5

    ###
    #
    # K-fold version
    #
    # Note: minDCF of calibrated folds may change with respect to the one we computed at the beginning over the whole dataset, since we are pooling scores of different folds that have undergone a different affine transformation
    #
    # Pay attention that, for fusion and model comparison, we need the folds to be the same across the two systems
    #
    # We use K = 5 (KFOLD variable)
    #
    ###
    
    labels =  LVAL
    # We start with the computation of the system performance on the calibration set (whole dataset)
    print('GMM: minDCF (0.1) = %.3f - actDCF (0.1) = %.3f' % (
        bayesRisk.compute_minDCF_binary_fast(SLLR_gmm, labels, 0.1, 1.0, 1.0),
        bayesRisk.compute_actDCF_binary_fast(SLLR_gmm, labels, 0.1, 1.0, 1.0)))

    print('SVM: minDCF (0.1) = %.3f - actDCF (0.1) = %.3f' % (
        bayesRisk.compute_minDCF_binary_fast(SVAL_svm, labels, 0.1, 1.0, 1.0),
        bayesRisk.compute_actDCF_binary_fast(SVAL_svm, labels, 0.1, 1.0, 1.0)))
    
    print('LR: minDCF (0.1) = %.3f - actDCF (0.1) = %.3f' % (
        bayesRisk.compute_minDCF_binary_fast(SVAL_lr, labels, 0.1, 1.0, 1.0),
        bayesRisk.compute_actDCF_binary_fast(SVAL_lr, labels, 0.1, 1.0, 1.0)))

    # Comparison of actDCF / minDCF of all systems
    logOdds, actDCF, minDCF = bayesPlot(SLLR_gmm, labels)
    axes[0,0].plot(logOdds, minDCF, color='C0', linestyle='--', label = 'minDCF')
    axes[0,0].plot(logOdds, actDCF, color='C0', linestyle='-', label = 'actDCF')

    logOdds, actDCF, minDCF = bayesPlot(SVAL_svm, labels)
    axes[1,0].plot(logOdds, minDCF, color='C1', linestyle='--', label = 'minDCF')
    axes[1,0].plot(logOdds, actDCF, color='C1', linestyle='-', label = 'actDCF')
    
    logOdds, actDCF, minDCF = bayesPlot(SVAL_lr, labels)
    axes[2,0].plot(logOdds, minDCF, color='C2', linestyle='--', label = 'minDCF')
    axes[2,0].plot(logOdds, actDCF, color='C2', linestyle='-', label = 'actDCF')
    
    axes[0,0].set_ylim(0, 0.8)    
    axes[0,0].legend()

    axes[1,0].set_ylim(0, 0.8)    
    axes[1,0].legend()
    
    axes[2,0].set_ylim(0, 0.8)    
    axes[2,0].legend()
    
    axes[0,0].set_title('GMM - validation - non-calibrated scores')
    axes[1,0].set_title('SVM - validation - non-calibrated scores')
    axes[2,0].set_title('LR - validation - non-calibrated scores')
    
    # We calibrate all systems (independently)
    
    PT = 0.1
    
    # GMM model calibration with kfold
    calibrated_scores_GMM = [] # We will add to the list the scores computed for each fold
    labels_GMM = [] # We need to ensure that we keep the labels aligned with the scores. The simplest thing to do is to just extract each fold label and pool all the fold labels together in the same order as we pool the corresponding scores.

    # We plot the non-calibrated minDCF and actDCF for reference
    logOdds, actDCF, minDCF = bayesPlot(SLLR_gmm, labels)
    axes[0,1].plot(logOdds, minDCF, color='C0', linestyle='--', label = 'minDCF (pre-cal.)')
    axes[0,1].plot(logOdds, actDCF, color='C0', linestyle=':', label = 'actDCF (pre-cal.)')
    print ('GMM')
    print ('\tValidation set')
    print ('\t\tminDCF(p=0.1), no cal.: %.3f' % bayesRisk.compute_minDCF_binary_fast(SLLR_gmm, labels, 0.1, 1.0, 1.0)) # Calibration may change minDCF due to being fold-dependent (thus it's not globally affine anymore)
    print ('\t\tactDCF(p=0.1), no cal.: %.3f' % bayesRisk.compute_actDCF_binary_fast(SLLR_gmm, labels, 0.1, 1.0, 1.0))
    
    # We train the calibration model for the prior pT = 0.2
    pT = PT
    # Train KFOLD times the calibration model
    for foldIdx in range(KFOLD):
        # keep 1 fold for validation, use the remaining ones for training
        SCAL, SVAL = extract_train_val_folds_from_ary(SLLR_gmm, foldIdx)
        LCAL, LVAL = extract_train_val_folds_from_ary(labels, foldIdx)
        # Train the model on the KFOLD - 1 training folds
        w, b = logReg.trainWeightedLogRegBinary(vrow(SCAL), LCAL, 0, pT)
        # Apply the model to the validation fold
        calibrated_SVAL =  (w.T @ vrow(SVAL) + b - numpy.log(pT / (1-pT))).ravel()
        # Add the scores of this validation fold to the cores list
        calibrated_scores_GMM.append(calibrated_SVAL)
        # Add the corresponding labels to preserve alignment between scores and labels
        labels_GMM.append(LVAL)

    # Re-build the score and label arrays (pooling) - these contains an entry for every element in the original dataset (but the order of the samples is different)
    calibrated_scores_GMM = numpy.hstack(calibrated_scores_GMM)
    labels_GMM = numpy.hstack(labels_GMM)

    # Evaluate the performance on pooled scores - we need to use the label vector labels_sys_1 since it's aligned to calibrated_scores_GMM    
    print ('\t\tminDCF(p=0.1), cal.   : %.3f' % bayesRisk.compute_minDCF_binary_fast(calibrated_scores_GMM, labels_GMM, 0.1, 1.0, 1.0)) # Calibration may change minDCF due to being fold-dependent (thus it's not globally affine anymore)
    print ('\t\tactDCF(p=0.1), cal.   : %.3f' % bayesRisk.compute_actDCF_binary_fast(calibrated_scores_GMM, labels_GMM, 0.1, 1.0, 1.0))
    
    logOdds, actDCF, minDCF = bayesPlot(calibrated_scores_GMM, labels_GMM)
    axes[0,1].plot(logOdds, actDCF, color='C0', linestyle='-', label = 'actDCF (cal.)') # NOTE: actDCF of the calibrated pooled scores MAY be lower than the global minDCF we computed earlier, since ache fold is calibrated on its own (thus it's as if we were estimating a possibly different threshold for each fold, whereas minDCF employs a single threshold for all scores)
    axes[0,1].legend()

    axes[0,1].set_title('GMM - validation')
    axes[0,1].set_ylim(0, 0.8)    
    
    # For K-fold the final model is a new model re-trained over the whole set, using the optimal hyperparameters we selected during the k-fold procedure (in this case we have no hyperparameter, so we simply train a new model on the whole dataset)

    w, b = logReg.trainWeightedLogRegBinary(vrow(SLLR_gmm), labels, 0, pT)

    # We can use the trained model for application / evaluation data
    calibrated_eval_scores_GMM = (w.T @ vrow(eval_scores_GMM) + b - numpy.log(pT / (1-pT))).ravel()

    print ('\tEvaluation set')
    print ('\t\tminDCF(p=0.1)         : %.3f' % bayesRisk.compute_minDCF_binary_fast(eval_scores_GMM, L_eval, 0.1, 1.0, 1.0))
    print ('\t\tactDCF(p=0.1), no cal.: %.3f' % bayesRisk.compute_actDCF_binary_fast(eval_scores_GMM, L_eval, 0.1, 1.0, 1.0))
    print ('\t\tactDCF(p=0.1), cal.   : %.3f' % bayesRisk.compute_actDCF_binary_fast(calibrated_eval_scores_GMM, L_eval, 0.1, 1.0, 1.0))    
    
    # We plot minDCF, non-calibrated DCF and calibrated DCF for system 1
    logOdds, actDCF_precal, minDCF = bayesPlot(eval_scores_GMM, L_eval)
    logOdds, actDCF_cal, _ = bayesPlot(calibrated_eval_scores_GMM, L_eval) # minDCF is the same
    axes[0,2].plot(logOdds, minDCF, color='C0', linestyle='--', label = 'minDCF')
    axes[0,2].plot(logOdds, actDCF_precal, color='C0', linestyle=':', label = 'actDCF (pre-cal.)')
    axes[0,2].plot(logOdds, actDCF_cal, color='C0', linestyle='-', label = 'actDCF (cal.)')
    axes[0,2].set_ylim(0.0, 0.8)
    axes[0,2].set_title('GMM - evaluation')
    axes[0,2].legend()
    

    
    # SVM
    calibrated_scores_SVM = [] # We will add to the list the scores computed for each fold
    labels_SVM = [] # We need to ensure that we keep the labels aligned with the scores. The simplest thing to do is to just extract each fold label and pool all the fold labels together in the same order as we pool the corresponding scores.

    scores_SVM = SVAL_svm 
    # We plot the non-calibrated minDCF and actDCF for reference
    logOdds, actDCF, minDCF = bayesPlot(scores_SVM, labels)
    axes[1,1].plot(logOdds, minDCF, color='C1', linestyle='--', label = 'minDCF (pre-cal.)')
    axes[1,1].plot(logOdds, actDCF, color='C1', linestyle=':', label = 'actDCF (pre-cal.)')
    print ('SVM')
    print ('\tValidation set')
    print ('\t\tminDCF(p=0.1), no cal.: %.3f' % bayesRisk.compute_minDCF_binary_fast(scores_SVM, labels, 0.1, 1.0, 1.0)) # Calibration may change minDCF due to being fold-dependent (thus it's not globally affine anymore)
    print ('\t\tactDCF(p=0.1), no cal.: %.3f' % bayesRisk.compute_actDCF_binary_fast(scores_SVM, labels, 0.1, 1.0, 1.0))
    
    # We train the calibration model for the prior pT = 0.2
    pT = PT
    # Train KFOLD times the calibration model
    for foldIdx in range(KFOLD):
        # keep 1 fold for validation, use the remaining ones for training
        SCAL, SVAL = extract_train_val_folds_from_ary(scores_SVM, foldIdx)
        LCAL, LVAL = extract_train_val_folds_from_ary(labels, foldIdx)
        # Train the model on the KFOLD - 1 training folds
        w, b = logReg.trainWeightedLogRegBinary(vrow(SCAL), LCAL, 0, pT)
        # Apply the model to the validation fold
        calibrated_SVAL =  (w.T @ vrow(SVAL) + b - numpy.log(pT / (1-pT))).ravel()
        # Add the scores of this validation fold to the cores list
        calibrated_scores_SVM.append(calibrated_SVAL)
        # Add the corresponding labels to preserve alignment between scores and labels
        labels_SVM.append(LVAL)

    # Re-build the score and label arrays (pooling) - these contains an entry for every element in the original dataset (but the order of the samples is different)
    calibrated_scores_SVM = numpy.hstack(calibrated_scores_SVM)
    labels_SVM = numpy.hstack(labels_SVM)

    # Evaluate the performance on pooled scores - we need to use the label vector labels_sys_2 since it's aligned to calibrated_scores_sys_2    
    print ('\t\tminDCF(p=0.1), cal.   : %.3f' % bayesRisk.compute_minDCF_binary_fast(calibrated_scores_SVM, labels_SVM, 0.1, 1.0, 1.0)) # Calibration may change minDCF due to being fold-dependent (thus it's not globally affine anymore)
    print ('\t\tactDCF(p=0.1), cal.   : %.3f' % bayesRisk.compute_actDCF_binary_fast(calibrated_scores_SVM, labels_SVM, 0.1, 1.0, 1.0))
    
    logOdds, actDCF, minDCF = bayesPlot(calibrated_scores_SVM, labels_SVM)
    axes[1,1].plot(logOdds, actDCF, color='C1', linestyle='-', label = 'actDCF (cal.)') # NOTE: actDCF of the calibrated pooled scores MAY be lower than the global minDCF we computed earlier, since ache fold is calibrated on its own (thus it's as if we were estimating a possibly different threshold for each fold, whereas minDCF employs a single threshold for all scores)
    axes[1,1].legend()

    axes[1,1].set_ylim(0, 0.8)            
    axes[1,1].set_title('SVM - validation')
    
    # For K-fold the final model is a new model re-trained over the whole set, using the optimal hyperparameters we selected during the k-fold procedure (in this case we have no hyperparameter, so we simply train a new model on the whole dataset)

    w, b = logReg.trainWeightedLogRegBinary(vrow(scores_SVM), labels, 0, pT)

    # We can use the trained model for application / evaluation data
    calibrated_eval_scores_SVM = (w.T @ vrow(eval_scores_SVM) + b - numpy.log(pT / (1-pT))).ravel()

    print ('\tEvaluation set')
    print ('\t\tminDCF(p=0.1)         : %.3f' % bayesRisk.compute_minDCF_binary_fast(eval_scores_SVM, L_eval, 0.1, 1.0, 1.0))
    print ('\t\tactDCF(p=0.1), no cal.: %.3f' % bayesRisk.compute_actDCF_binary_fast(eval_scores_SVM, L_eval, 0.1, 1.0, 1.0))
    print ('\t\tactDCF(p=0.1), cal.   : %.3f' % bayesRisk.compute_actDCF_binary_fast(calibrated_eval_scores_SVM, L_eval, 0.1, 1.0, 1.0))    
    
    # We plot minDCF, non-calibrated DCF and calibrated DCF for system 1
    logOdds, actDCF_precal, minDCF = bayesPlot(eval_scores_SVM, L_eval)
    logOdds, actDCF_cal, _ = bayesPlot(calibrated_eval_scores_SVM, L_eval) # minDCF is the same
    axes[1,2].plot(logOdds, minDCF, color='C1', linestyle='--', label = 'minDCF')
    axes[1,2].plot(logOdds, actDCF_precal, color='C1', linestyle=':', label = 'actDCF (pre-cal.)')
    axes[1,2].plot(logOdds, actDCF_cal, color='C1', linestyle='-', label = 'actDCF (cal.)')
    axes[1,2].set_ylim(0.0, 0.8)
    axes[1,2].set_title('SVM - evaluation')
    axes[1,2].legend()

    



    # Loggistic Regression
    calibrated_scores_LR = [] # We will add to the list the scores computed for each fold
    labels_LR = [] # We need to ensure that we keep the labels aligned with the scores. The simplest thing to do is to just extract each fold label and pool all the fold labels together in the same order as we pool the corresponding scores.

    scores_LR = SVAL_lr 
    # We plot the non-calibrated minDCF and actDCF for reference
    logOdds, actDCF, minDCF = bayesPlot(scores_LR, labels)
    axes[2,1].plot(logOdds, minDCF, color='C2', linestyle='--', label = 'minDCF (pre-cal.)')
    axes[2,1].plot(logOdds, actDCF, color='C2', linestyle=':', label = 'actDCF (pre-cal.)')
    print ('Logistic regression')
    print ('\tValidation set')
    print ('\t\tminDCF(p=0.1), no cal.: %.3f' % bayesRisk.compute_minDCF_binary_fast(scores_LR, labels, 0.1, 1.0, 1.0)) # Calibration may change minDCF due to being fold-dependent (thus it's not globally affine anymore)
    print ('\t\tactDCF(p=0.1), no cal.: %.3f' % bayesRisk.compute_actDCF_binary_fast(scores_LR, labels, 0.1, 1.0, 1.0))
    
    # We train the calibration model for the prior pT = 0.2
    pT = PT
    # Train KFOLD times the calibration model
    for foldIdx in range(KFOLD):
        # keep 1 fold for validation, use the remaining ones for training
        SCAL, SVAL = extract_train_val_folds_from_ary(scores_LR, foldIdx)
        LCAL, LVAL = extract_train_val_folds_from_ary(labels, foldIdx)
        # Train the model on the KFOLD - 1 training folds
        w, b = logReg.trainWeightedLogRegBinary(vrow(SCAL), LCAL, 0, pT)
        # Apply the model to the validation fold
        calibrated_SVAL =  (w.T @ vrow(SVAL) + b - numpy.log(pT / (1-pT))).ravel()
        # Add the scores of this validation fold to the cores list
        calibrated_scores_LR.append(calibrated_SVAL)
        # Add the corresponding labels to preserve alignment between scores and labels
        labels_LR.append(LVAL)

    # Re-build the score and label arrays (pooling) - these contains an entry for every element in the original dataset (but the order of the samples is different)
    calibrated_scores_LR = numpy.hstack(calibrated_scores_LR)
    labels_LR = numpy.hstack(labels_LR)

    # Evaluate the performance on pooled scores - we need to use the label vector labels_sys_2 since it's aligned to calibrated_scores_sys_2    
    print ('\t\tminDCF(p=0.1), cal.   : %.3f' % bayesRisk.compute_minDCF_binary_fast(calibrated_scores_LR, labels_LR, 0.1, 1.0, 1.0)) # Calibration may change minDCF due to being fold-dependent (thus it's not globally affine anymore)
    print ('\t\tactDCF(p=0.1), cal.   : %.3f' % bayesRisk.compute_actDCF_binary_fast(calibrated_scores_LR, labels_LR, 0.1, 1.0, 1.0))
    
    logOdds, actDCF, minDCF = bayesPlot(calibrated_scores_LR, labels_LR)
    axes[2,1].plot(logOdds, actDCF, color='C2', linestyle='-', label = 'actDCF (cal.)') # NOTE: actDCF of the calibrated pooled scores MAY be lower than the global minDCF we computed earlier, since ache fold is calibrated on its own (thus it's as if we were estimating a possibly different threshold for each fold, whereas minDCF employs a single threshold for all scores)
    axes[2,1].legend()

    axes[2,1].set_ylim(0, 0.8)            
    axes[2,1].set_title('LR - validation')
    
    # For K-fold the final model is a new model re-trained over the whole set, using the optimal hyperparameters we selected during the k-fold procedure (in this case we have no hyperparameter, so we simply train a new model on the whole dataset)

    w, b = logReg.trainWeightedLogRegBinary(vrow(scores_LR), labels, 0, pT)

    # We can use the trained model for application / evaluation data
    calibrated_eval_scores_LR = (w.T @ vrow(eval_scores_LR) + b - numpy.log(pT / (1-pT))).ravel()

    print ('\tEvaluation set')
    print ('\t\tminDCF(p=0.1)         : %.3f' % bayesRisk.compute_minDCF_binary_fast(eval_scores_LR, L_eval, 0.1, 1.0, 1.0))
    print ('\t\tactDCF(p=0.1), no cal.: %.3f' % bayesRisk.compute_actDCF_binary_fast(eval_scores_LR, L_eval, 0.1, 1.0, 1.0))
    print ('\t\tactDCF(p=0.1), cal.   : %.3f' % bayesRisk.compute_actDCF_binary_fast(calibrated_eval_scores_LR, L_eval, 0.1, 1.0, 1.0))    
    
    # We plot minDCF, non-calibrated DCF and calibrated DCF for system 1
    logOdds, actDCF_precal, minDCF = bayesPlot(eval_scores_LR, L_eval)
    logOdds, actDCF_cal, _ = bayesPlot(calibrated_eval_scores_LR, L_eval) # minDCF is the same
    axes[2,2].plot(logOdds, minDCF, color='C2', linestyle='--', label = 'minDCF')
    axes[2,2].plot(logOdds, actDCF_precal, color='C2', linestyle=':', label = 'actDCF (pre-cal.)')
    axes[2,2].plot(logOdds, actDCF_cal, color='C2', linestyle='-', label = 'actDCF (cal.)')
    axes[2,2].set_ylim(0.0, 0.8)
    axes[2,2].set_title('LR - evaluation')
    axes[2,2].legend()
    
    
    
    
    
    # Fusion #
    
    fusedScores = [] # We will add to the list the scores computed for each fold
    fusedLabels = [] # We need to ensure that we keep the labels aligned with the scores. The simplest thing to do is to just extract each fold label and pool all the fold labels together in the same order as we pool the corresponding scores.
    
    # We train the fusion for the prior pT = 0.2
    pT = PT
    
    # Train KFOLD times the fusion model
    for foldIdx in range(KFOLD):
        # keep 1 fold for validation, use the remaining ones for training        
        SCAL1, SVAL1 = extract_train_val_folds_from_ary(SLLR_gmm, foldIdx)
        SCAL2, SVAL2 = extract_train_val_folds_from_ary(SVAL_svm, foldIdx)
        SCAL3, SVAL3 = extract_train_val_folds_from_ary(SVAL_lr, foldIdx)
        LCAL, LVAL = extract_train_val_folds_from_ary(labels, foldIdx)
        # Build the training scores "feature" matrix
        SCAL = numpy.vstack([SCAL1, SCAL2, SCAL3])
        # Train the model on the KFOLD - 1 training folds
        w, b = logReg.trainWeightedLogRegBinary(SCAL, LCAL, 0, pT)
        # Build the validation scores "feature" matrix
        SVAL = numpy.vstack([SVAL1, SVAL2, SVAL3])
        # Apply the model to the validation fold
        calibrated_SVAL =  (w.T @ SVAL + b - numpy.log(pT / (1-pT))).ravel()
        # Add the scores of this validation fold to the cores list
        fusedScores.append(calibrated_SVAL)
        # Add the corresponding labels to preserve alignment between scores and labels
        fusedLabels.append(LVAL)

    # Re-build the score and label arrays (pooling) - these contains an entry for every element in the original dataset (but the order of the samples is different)        
    fusedScores = numpy.hstack(fusedScores)
    fusedLabels = numpy.hstack(fusedLabels)

    # Evaluate the performance on pooled scores - we need to use the label vector fusedLabels since it's aligned to calScores_sys_2 (plot on same figure as system 1 and system 2)

    print ('Fusion')
    print ('\tValidation set')
    print ('\t\tminDCF(p=0.1)         : %.3f' % bayesRisk.compute_minDCF_binary_fast(fusedScores, fusedLabels, 0.1, 1.0, 1.0)) # Calibration may change minDCF due to being fold-dependent (thus it's not globally affine anymore)
    print ('\t\tactDCF(p=0.1)         : %.3f' % bayesRisk.compute_actDCF_binary_fast(fusedScores, fusedLabels, 0.1, 1.0, 1.0))

    # As comparison, we select calibrated models trained with prior 0.2 (our target application)
    logOdds, actDCF, minDCF = bayesPlot(calibrated_scores_GMM, labels_GMM)
    axes[3,1].set_title('Fusion - validation')
    axes[3,1].plot(logOdds, minDCF, color='C0', linestyle='--', label = 'S1 - minDCF')
    axes[3,1].plot(logOdds, actDCF, color='C0', linestyle='-', label = 'S1 - actDCF')
    logOdds, actDCF, minDCF = bayesPlot(calibrated_scores_SVM, labels_SVM)
    axes[3,1].plot(logOdds, minDCF, color='C1', linestyle='--', label = 'S2 - minDCF')
    axes[3,1].plot(logOdds, actDCF, color='C1', linestyle='-', label = 'S2 - actDCF')    
    logOdds, actDCF, minDCF = bayesPlot(calibrated_scores_LR, labels_LR)
    axes[3,1].plot(logOdds, minDCF, color='C2', linestyle='--', label = 'S2 - minDCF')
    axes[3,1].plot(logOdds, actDCF, color='C2', linestyle='-', label = 'S2 - actDCF')
    
    logOdds, actDCF, minDCF = bayesPlot(fusedScores, fusedLabels)
    axes[3,1].plot(logOdds, minDCF, color='C3', linestyle='--', label = 'S1 + S2 + S3 - KFold - minDCF(0.1)')
    axes[3,1].plot(logOdds, actDCF, color='C3', linestyle='-', label = 'S1 + S2 + S3 - KFold - actDCF(0.1)')
    axes[3,1].set_ylim(0.0, 0.8)
    axes[3,1].legend()





    # For K-fold the final model is a new model re-trained over the whole set, using the optimal hyperparameters we selected during the k-fold procedure (in this case we have no hyperparameter, so we simply train a new model on the whole dataset)

    SMatrix = numpy.vstack([SLLR_gmm, SVAL_svm, SVAL_lr])
    w, b = logReg.trainWeightedLogRegBinary(SMatrix, labels, 0, pT)

    # Apply model to application / evaluation data
    SMatrixEval = numpy.vstack([eval_scores_GMM, eval_scores_SVM, eval_scores_LR])
    fused_eval_scores = (w.T @ SMatrixEval + b - numpy.log(pT / (1-pT))).ravel()

    print ('\tEvaluation set')
    print ('\t\tminDCF(p=0.1)         : %.3f' % bayesRisk.compute_minDCF_binary_fast(fused_eval_scores, L_eval, 0.1, 1.0, 1.0))
    print ('\t\tactDCF(p=0.1)         : %.3f' % bayesRisk.compute_actDCF_binary_fast(fused_eval_scores, L_eval, 0.1, 1.0, 1.0))
    
    # We plot minDCF, actDCF for calibrated system 1, calibrated system 2 and fusion
    logOdds, actDCF, minDCF = bayesPlot(calibrated_eval_scores_GMM, L_eval)
    axes[3,2].plot(logOdds, minDCF, color='C0', linestyle='--', label = 'S1 - minDCF')
    axes[3,2].plot(logOdds, actDCF, color='C0', linestyle='-', label = 'S1 - actDCF')
    logOdds, actDCF, minDCF = bayesPlot(calibrated_eval_scores_SVM, L_eval)
    axes[3,2].plot(logOdds, minDCF, color='C1', linestyle='--', label = 'S2 - minDCF')
    axes[3,2].plot(logOdds, actDCF, color='C1', linestyle='-', label = 'S2 - actDCF')
    logOdds, actDCF, minDCF = bayesPlot(calibrated_eval_scores_LR, L_eval)
    axes[3,2].plot(logOdds, minDCF, color='C2', linestyle='--', label = 'S3 - minDCF')
    axes[3,2].plot(logOdds, actDCF, color='C2', linestyle='-', label = 'S3 - actDCF')
    
    logOdds, actDCF, minDCF = bayesPlot(fused_eval_scores, L_eval) # minDCF is the same
    axes[3,2].plot(logOdds, minDCF, color='C3', linestyle='--', label = 'S1 + S2 + S3 - minDCF')
    axes[3,2].plot(logOdds, actDCF, color='C3', linestyle='-', label = 'S1 + S2 + S3 - actDCF')
    axes[3,2].set_ylim(0.0, 0.8)
    axes[3,2].set_title('Fusion - evaluation')
    axes[3,2].legend()
    
    plt.show()
    
    
    plt.figure(figsize=(12, 6))
    bayes_error_plot(minDCF, actDCF, logOdds, 'delivered model, fusion model')
    plt.show()
    
    
    threshold = 0
    predictions = fused_eval_scores > threshold
    
    # Step 2: Calculate accuracy
    accuracy = numpy.mean(predictions == L_eval) * 100
    print(f'Accuracy on the evaluation set: {accuracy:.2f}%')
    
    
    
    
    lambdas = numpy.logspace(-4, 2, 13)
    preprocessing_methods = ['none', 'center', 'znormalize', 'whiten', 'expand']
    
    for method in preprocessing_methods:
        DTR_preprocessed = preprocess_data(DTR, method)
        DVAL_preprocessed = preprocess_data(DVAL, method)
        D_eval_preprocessed = preprocess_data(D_eval, method)
        
        minDCFs = []
        actDCFs = []
        
        for lamb in lambdas:
            print('lambda= ', lamb)
            w, b = trainLogRegBinary(DTR_preprocessed, LTR, lamb)  # Train model
            
            # Compute evaluation scores
            sEval = numpy.dot(w.T, D_eval_preprocessed) + b
            P_eval = (sEval > 0) * 1  # Predict evaluation labels
            
            # Compute empirical prior
            pEmp = (LTR == 1).sum() / LTR.size
            
            # Compute LLR-like scores
            sEvalLLR = sEval - numpy.log(pEmp / (1 - pEmp))
            
            # Compute optimal decisions for the three priors 0.1
            pT = 0.1
            minDCF = bayesRisk.compute_minDCF_binary_fast(sEvalLLR, L_eval, pT, 1.0, 1.0)
            actDCF = bayesRisk.compute_actDCF_binary_fast(sEvalLLR, L_eval, pT, 1.0, 1.0)
            
            minDCFs.append(minDCF)
            actDCFs.append(actDCF)
            
            print(f'{method} - lambda = {lamb:.1e} - minDCF: {minDCF:.4f}')
            
            print()