import numpy
import scipy.special
import matplotlib
import matplotlib.pyplot

import GaussianModels
import Ploting
import pca
# import lda

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

# compute matrix of posteriors from class-conditional log-likelihoods (each column represents a sample) and prior array
def compute_posteriors(log_clas_conditional_ll, prior_array):
    logJoint = log_clas_conditional_ll + vcol(numpy.log(prior_array))
    logPost = logJoint - scipy.special.logsumexp(logJoint, 0)
    return numpy.exp(logPost)

# Compute optimal Bayes decisions for the matrix of class posterior (each column refers to a sample)
def compute_optimal_Bayes(posterior, costMatrix):
    expectedCosts = costMatrix @ posterior
    return numpy.argmin(expectedCosts, 0)

# Build uniform cost matrix with cost 1 for all kinds of error, and cost 0 for correct assignments
def uniform_cost_matrix(nClasses):
    return numpy.ones((nClasses, nClasses)) - numpy.eye(nClasses)

# Assume that classes are labeled 0, 1, 2 ... (nClasses - 1)
def compute_confusion_matrix(predictedLabels, classLabels):
    nClasses = classLabels.max() + 1
    M = numpy.zeros((nClasses, nClasses), dtype=numpy.int32)
    for i in range(classLabels.size):
        M[predictedLabels[i], classLabels[i]] += 1
    return M

# Optimal Bayes deicsions for binary tasks with log-likelihood-ratio scores
def compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp):
    th = -numpy.log( (prior * Cfn) / ((1 - prior) * Cfp) )
    return numpy.int32(llr > th)

# Multiclass solution that works also for binary problems
def compute_empirical_Bayes_risk(predictedLabels, classLabels, prior_array, costMatrix, normalize=True):
    M = compute_confusion_matrix(predictedLabels, classLabels) # Confusion matrix
    errorRates = M / vrow(M.sum(0))
    bayesError = ((errorRates * costMatrix).sum(0) * prior_array.ravel()).sum()
    if normalize:
        return bayesError / numpy.min(costMatrix @ vcol(prior_array))
    return bayesError

# Specialized function for binary problems (empirical_Bayes_risk is also called DCF or actDCF)
def compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp, normalize=True):
    M = compute_confusion_matrix(predictedLabels, classLabels) # Confusion matrix
    Pfn = M[0,1] / (M[0,1] + M[1,1])
    Pfp = M[1,0] / (M[0,0] + M[1,0])
    bayesError = prior * Cfn * Pfn + (1-prior) * Cfp * Pfp
    if normalize:
        return bayesError / numpy.minimum(prior * Cfn, (1-prior)*Cfp)
    return bayesError

# Compute empirical Bayes (DCF or actDCF) risk from llr with optimal Bayes decisions
def compute_empirical_Bayes_risk_binary_llr_optimal_decisions(llr, classLabels, prior, Cfn, Cfp, normalize=True):
    predictedLabels = compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp)
    return compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp, normalize=normalize)

# Compute all combinations of Pfn, Pfp for all thresholds (sorted)
def compute_Pfn_Pfp_allThresholds_slow(llr, classLabels):
    llrSorter = numpy.argsort(llr)
    llrSorted = llr[llrSorter] # We sort the llrs

    Pfn = []
    Pfp = []
    thresholds = numpy.concatenate([numpy.array([-numpy.inf]), llrSorted, numpy.array([numpy.inf])]) #The function returns a slightly different array than the fast version, which does not include -numpy.inf as threshold - see the fast function comment
    for th in thresholds:
        M = compute_confusion_matrix(predictedLabels, classLabels) # Confusion matrix
        Pfn.append(M[0,1] / (M[0,1] + M[1,1]))
        Pfp.append(M[1,0] / (M[0,0] + M[1,0]))
    return Pfn, Pfp, thresholds
        
    
    
# Compute minDCF (slow version, loop over all thresholds recomputing the costs)
# Note: for minDCF llrs can be arbitrary scores, since we are optimizing the threshold
# We can therefore directly pass the logistic regression scores, or the SVM scores
def compute_minDCF_binary_slow(llr, classLabels, prior, Cfn, Cfp, returnThreshold=False):
    # llrSorter = numpy.argsort(llr) 
    # llrSorted = llr[llrSorter] # We sort the llrs
    # classLabelsSorted = classLabels[llrSorter] # we sort the labels so that they are aligned to the llrs
    # We can remove this part
    llrSorted = llr # In this function (slow version) sorting is not really necessary, since we re-compute the predictions and confusion matrices everytime
    
    thresholds = numpy.concatenate([numpy.array([-numpy.inf]), llrSorted, numpy.array([numpy.inf])])
    dcfMin = None
    dcfTh = None
    for th in thresholds:
        predictedLabels = numpy.int32(llr > th)
        dcf = compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp)
        if dcfMin is None or dcf < dcfMin:
            dcfMin = dcf
            dcfTh = th
    if returnThreshold:
        return dcfMin, dcfTh
    else:
        return dcfMin

# Compute minDCF (fast version)
# If we sort the scores, then, as we sweep the scores, we can have that at most one prediction changes everytime. We can then keep a running confusion matrix (or simply the number of false positives and false negatives) that is updated everytime we move the threshold

# Auxiliary function, returns all combinations of Pfp, Pfn corresponding to all possible thresholds
# We do not consider -inf as threshld, since we use as assignment llr > th, so the left-most score corresponds to all samples assigned to class 1 already
def compute_Pfn_Pfp_allThresholds_fast(llr, classLabels):
    llrSorter = numpy.argsort(llr)
    llrSorted = llr[llrSorter] # We sort the llrs
    classLabelsSorted = classLabels[llrSorter] # we sort the labels so that they are aligned to the llrs

    Pfp = []
    Pfn = []
    
    nTrue = (classLabelsSorted==1).sum()
    nFalse = (classLabelsSorted==0).sum()
    nFalseNegative = 0 # With the left-most theshold all samples are assigned to class 1
    nFalsePositive = nFalse
    
    Pfn.append(nFalseNegative / nTrue)
    Pfp.append(nFalsePositive / nFalse)
    
    for idx in range(len(llrSorted)):
        if classLabelsSorted[idx] == 1:
            nFalseNegative += 1 # Increasing the threshold we change the assignment for this llr from 1 to 0, so we increase the error rate
        if classLabelsSorted[idx] == 0:
            nFalsePositive -= 1 # Increasing the threshold we change the assignment for this llr from 1 to 0, so we decrease the error rate
        Pfn.append(nFalseNegative / nTrue)
        Pfp.append(nFalsePositive / nFalse)

    #The last values of Pfn and Pfp should be 1.0 and 0.0, respectively
    #Pfn.append(1.0) # Corresponds to the numpy.inf threshold, all samples are assigned to class 0
    #Pfp.append(0.0) # Corresponds to the numpy.inf threshold, all samples are assigned to class 0
    llrSorted = numpy.concatenate([-numpy.array([numpy.inf]), llrSorted])

    # In case of repeated scores, we need to "compact" the Pfn and Pfp arrays (i.e., we need to keep only the value that corresponds to an actual change of the threshold
    PfnOut = []
    PfpOut = []
    thresholdsOut = []
    for idx in range(len(llrSorted)):
        if idx == len(llrSorted) - 1 or llrSorted[idx+1] != llrSorted[idx]: # We are indeed changing the threshold, or we have reached the end of the array of sorted scores
            PfnOut.append(Pfn[idx])
            PfpOut.append(Pfp[idx])
            thresholdsOut.append(llrSorted[idx])
            
    return numpy.array(PfnOut), numpy.array(PfpOut), numpy.array(thresholdsOut) # we return also the corresponding thresholds
    
# Note: for minDCF llrs can be arbitrary scores, since we are optimizing the threshold
# We can therefore directly pass the logistic regression scores, or the SVM scores
def compute_minDCF_binary_fast(llr, classLabels, prior, Cfn, Cfp, returnThreshold=False):

    Pfn, Pfp, th = compute_Pfn_Pfp_allThresholds_fast(llr, classLabels)
    minDCF = (prior * Cfn * Pfn + (1 - prior) * Cfp * Pfp) / numpy.minimum(prior * Cfn, (1-prior)*Cfp) # We exploit broadcasting to compute all DCFs for all thresholds
    idx = numpy.argmin(minDCF)
    if returnThreshold:
        return minDCF[idx], th[idx]
    else:
        return minDCF[idx]

compute_actDCF_binary_fast = compute_empirical_Bayes_risk_binary_llr_optimal_decisions # To have a function with a similar name to the minDCF one

def compute_llr_from_log_likelihood(S_logLikelihood):
    return S_logLikelihood[1, :] - S_logLikelihood[0, :]

def plot_bayes_error(llr, LVAL, title):
    """
    Plot the Bayes error curves for two different epsilon values.
    
    Parameters:
    - llr: numpy array of log-likelihood ratios for the binary classification
    - LVAL: numpy array of labels for the binary classification
   
    """

    # Define the effective prior log odds and compute effective priors
    effPriorLogOdds = numpy.linspace(-4, 4, 21)
    effPriors = 1.0 / (1.0 + numpy.exp(-effPriorLogOdds))
    
    # Initialize lists to store the DCF values
    actDCF = []
    minDCF = []

    # Compute actDCF and minDCF for eps = 0.001
    for effPrior in effPriors:
        commedia_predictions_binary = compute_optimal_Bayes_binary_llr(llr, effPrior, 1.0, 1.0)
        actDCF.append(compute_empirical_Bayes_risk_binary(commedia_predictions_binary, LVAL, effPrior, 1.0, 1.0))
        minDCF.append(compute_minDCF_binary_fast(llr, LVAL, effPrior, 1.0, 1.0))
    
    matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(effPriorLogOdds, actDCF, label='actDCF eps 0.001', color='r')
    matplotlib.pyplot.plot(effPriorLogOdds, minDCF, label='DCF eps 0.001', color='b')
    matplotlib.pyplot.ylim([0, 1.1])
    
    # Load commedia data
    #commedia_llr_binary = numpy.load(commedia_llr_path)
    #commedia_labels_binary = numpy.load(commedia_labels_path)

    # Compute actDCF and minDCF for eps = 1.0
    actDCF = []
    minDCF = []
    for effPrior in effPriors:
        commedia_predictions_binary = compute_optimal_Bayes_binary_llr(llr, effPrior, 1.0, 1.0)
        actDCF.append(compute_empirical_Bayes_risk_binary(commedia_predictions_binary, LVAL, effPrior, 1.0, 1.0))
        minDCF.append(compute_minDCF_binary_fast(llr, LVAL, effPrior, 1.0, 1.0))
    
    matplotlib.pyplot.plot(effPriorLogOdds, actDCF, label='actDCF eps 1.0', color='y')
    matplotlib.pyplot.plot(effPriorLogOdds, minDCF, label='DCF eps 1.0', color='c')
    matplotlib.pyplot.ylim([0, 1.1])

    matplotlib.pyplot.title(title)
    # Add legend and display the plot
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()
    
if __name__ == '__main__':

    D, L = Ploting.load('traindata.txt')
    (DTR, LTR), (DVAL, LVAL) = GaussianModels.split_db_2to1(D, L)

    
   
    for lab in [0,1]:
        D = DTR[:, LTR==lab] 
        mu, C = GaussianModels.compute_mu_C(D)
        
       
    hParams_MVG = GaussianModels.Gau_MVG_ML_estimates(DTR, LTR)
    S_logLikelihood = GaussianModels.compute_log_likelihood_Gau(DVAL, hParams_MVG) #hparam contains mu and C
    S_logPost = GaussianModels.compute_logPosterior(S_logLikelihood, numpy.ones(2)/2.)   
    # Predict labels
    PVAL = S_logPost.argmax(0)      
    print('confussion matrix- MVG:\n',compute_confusion_matrix(PVAL, LVAL)) #correct
    
    
    #Tied Guassian Classifier

   
    hParams_Tied = GaussianModels.Gau_Tied_ML_estimates(DTR, LTR)
        
    S_logLikelihood = GaussianModels.compute_log_likelihood_Gau(DVAL, hParams_Tied)
    S_logPost = GaussianModels.compute_logPosterior(S_logLikelihood, numpy.ones(2)/2.)
    PVAL = S_logPost.argmax(0)
    print('confussion matrix- tied: \n',compute_confusion_matrix(PVAL, LVAL))
   

    # Binary task
    print()
    print("-"*40)
    print()
    print("Binary task")
    S_logLikelihood = GaussianModels.compute_log_likelihood_Gau(DVAL, hParams_MVG)
    llr = compute_llr_from_log_likelihood(S_logLikelihood)

    for prior, Cfn, Cfp in [(0.5, 1, 1), (0.9, 1, 1), (0.1, 1, 1), (0.5, 1, 9), (0.5, 9, 1)]:
        print()
        print('Prior', prior, '- Cfn', Cfn, '- Cfp', Cfp)
        predictions_binary = compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp)
        
        print(compute_confusion_matrix(predictions_binary, LVAL))
        print('DCF (non-normalized): %.3f' % (compute_empirical_Bayes_risk_binary(
              predictions_binary, LVAL, prior, Cfn, Cfp, normalize=False)))
        print('DCF (normalized): %.3f' % (compute_empirical_Bayes_risk_binary(
            predictions_binary, LVAL, prior, Cfn, Cfp, normalize=True)))
        minDCF, minDCFThreshold = compute_minDCF_binary_fast(llr, LVAL, prior, Cfn, Cfp, returnThreshold=True)
        print('MinDCF (normalized, fast): %.3f (@ th = %e)' % (minDCF, minDCFThreshold))


    print()
    print("-"*40)
    print()

    print("MVG Model")
    for prior, Cfn, Cfp in [(0.1, 1, 1), (0.5, 1, 1), (0.9, 1, 1)]:
        print()
        print('Prior', prior, '- Cfn', Cfn, '- Cfp', Cfp)
        predictions_binary = compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp)
        print("Confusion Matrix:")
        print(compute_confusion_matrix(predictions_binary, LVAL))
        print('DCF (non-normalized): %.3f' % (compute_empirical_Bayes_risk_binary(
              predictions_binary, LVAL, prior, Cfn, Cfp, normalize=False)))
        print('DCF (normalized): %.3f' % (compute_empirical_Bayes_risk_binary(
            predictions_binary, LVAL, prior, Cfn, Cfp, normalize=True)))
        minDCF, minDCFThreshold = compute_minDCF_binary_fast(llr, LVAL, prior, Cfn, Cfp, returnThreshold=True)
        print('MinDCF (normalized, fast): %.3f (@ th = %e)' % (minDCF, minDCFThreshold))
    
    # PCA dimensionality reduction and model training
    for m in [2, 5, 10]:
        print(f"PCA with m = {m}")
        P = pca.compute_pca(DTR, m)
        DTR_PCA = pca.apply_pca(P, DTR)
        DVAL_PCA = pca.apply_pca(P, DVAL)

        # MVG with PCA
        hParams_MVG_PCA = GaussianModels.Gau_MVG_ML_estimates(DTR_PCA, LTR)
        S_logLikelihood = GaussianModels.compute_log_likelihood_Gau(DVAL_PCA, hParams_MVG_PCA)
        llr = compute_llr_from_log_likelihood(S_logLikelihood)
        
        print("MVG Model with PCA")
        for prior, Cfn, Cfp in [(0.1, 1, 1), (0.5, 1, 1), (0.9, 1, 1)]:
            print()
            print('Prior', prior, '- Cfn', Cfn, '- Cfp', Cfp)
            predictions_binary = compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp)
            print("Confusion Matrix:")
            print(compute_confusion_matrix(predictions_binary, LVAL))
            print('DCF (non-normalized): %.3f' % (compute_empirical_Bayes_risk_binary(
                  predictions_binary, LVAL, prior, Cfn, Cfp, normalize=False)))
            print('DCF (normalized): %.3f' % (compute_empirical_Bayes_risk_binary(
                predictions_binary, LVAL, prior, Cfn, Cfp, normalize=True)))
            minDCF, minDCFThreshold = compute_minDCF_binary_fast(llr, LVAL, prior, Cfn, Cfp, returnThreshold=True)
            print('MinDCF (normalized, fast): %.3f (@ th = %e)' % (minDCF, minDCFThreshold))



    #m=5
    m=5
    P = pca.compute_pca(DTR, m)
    DTR_PCA = pca.apply_pca(P, DTR)
    DVAL_PCA = pca.apply_pca(P, DVAL)
    
    
    
    # MVG with PCA
    hParams_MVG_PCA = GaussianModels.Gau_MVG_ML_estimates(DTR_PCA, LTR)
    S_logLikelihood = GaussianModels.compute_log_likelihood_Gau(DVAL_PCA, hParams_MVG_PCA)
    llr = compute_llr_from_log_likelihood(S_logLikelihood)
    
    plot_bayes_error(llr, LVAL, 'MVG')
    for prior, Cfn, Cfp in [(0.1, 1, 1), (0.5, 1, 1), (0.9, 1, 1)]:
        print('\n MVG with PCA and m=5')
        print('Prior', prior, '- Cfn', Cfn, '- Cfp', Cfp)
        minDCF, minDCFThreshold = compute_minDCF_binary_fast(llr, LVAL, prior, Cfn, Cfp, returnThreshold=True)
        print('MinDCF (normalized, fast): %.3f (@ th = %e)' % (minDCF, minDCFThreshold))
        
        
    #naive Bayes Guassian Classifier
    
    hParams_Naive = GaussianModels.Gau_Naive_ML_estimates(DTR, LTR)
    S_logLikelihood = GaussianModels.compute_log_likelihood_Gau(DVAL, hParams_Naive)
    llr = compute_llr_from_log_likelihood(S_logLikelihood)
    
    plot_bayes_error(llr, LVAL, 'naive bayes guassian')
    for prior, Cfn, Cfp in [(0.1, 1, 1), (0.5, 1, 1), (0.9, 1, 1)]:
        print('\n naive bayes with PCA and m=5')
        print('Prior', prior, '- Cfn', Cfn, '- Cfp', Cfp)
        minDCF, minDCFThreshold = compute_minDCF_binary_fast(llr, LVAL, prior, Cfn, Cfp, returnThreshold=True)
        print('MinDCF (normalized, fast): %.3f (@ th = %e)' % (minDCF, minDCFThreshold))
        
    #Tied Guassian Classifier

    hParams_Tied = GaussianModels.Gau_Tied_ML_estimates(DTR, LTR)
    S_logLikelihood = GaussianModels.compute_log_likelihood_Gau(DVAL, hParams_Tied)
    llr = compute_llr_from_log_likelihood(S_logLikelihood)
    
    plot_bayes_error(llr, LVAL, 'Tied guassian')
    for prior, Cfn, Cfp in [(0.1, 1, 1), (0.5, 1, 1), (0.9, 1, 1)]:
        print('\n tied with PCA and m=5')
        print('Prior', prior, '- Cfn', Cfn, '- Cfp', Cfp)
        minDCF, minDCFThreshold = compute_minDCF_binary_fast(llr, LVAL, prior, Cfn, Cfp, returnThreshold=True)
        print('MinDCF (normalized, fast): %.3f (@ th = %e)' % (minDCF, minDCFThreshold))

    
