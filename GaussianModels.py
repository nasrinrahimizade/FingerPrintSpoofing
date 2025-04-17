
import numpy
import scipy.special
import pca as lab3
import Ploting

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

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

def logpdf_GAU_ND(x, mu, C):
    P = numpy.linalg.inv(C)
    return -0.5*x.shape[0]*numpy.log(numpy.pi*2) - 0.5*numpy.linalg.slogdet(C)[1] - 0.5 * ((x-mu) * (P @ (x-mu))).sum(0)

def load_iris():
    
    import sklearn.datasets
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

# Compute a dictionary of ML parameters for each class
def Gau_MVG_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    for lab in labelSet:
        DX = D[:, L==lab]
        hParams[lab] = compute_mu_C(DX)
    return hParams

# Compute a dictionary of ML parameters for each class - Naive Bayes version of the model
# We compute the full covariance matrix and then extract the diagonal. Efficient implementations would work directly with just the vector of variances (diagonal of the covariance matrix)
def Gau_Naive_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    for lab in labelSet:
        DX = D[:, L==lab]
        mu, C = compute_mu_C(DX)
        hParams[lab] = (mu, C * numpy.eye(D.shape[0]))
    return hParams

# Compute a dictionary of ML parameters for each class - Tied Gaussian model
# We exploit the fact that the within-class covairance matrix is a weighted mean of the covraince matrices of the different classes
def Gau_Tied_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    hMeans = {}
    CGlobal = 0
    for lab in labelSet:
        DX = D[:, L==lab]
        mu, C_class = compute_mu_C(DX)
        CGlobal += C_class * DX.shape[1]
        hMeans[lab] = mu
    CGlobal = CGlobal / D.shape[1]
    for lab in labelSet:
        hParams[lab] = (hMeans[lab], CGlobal)
    return hParams

# Compute per-class log-densities. We assume classes are labeled from 0 to C-1. The parameters of each class are in hParams (for class i, hParams[i] -> (mean, cov))
def compute_log_likelihood_Gau(D, hParams):

    S = numpy.zeros((len(hParams), D.shape[1]))
    for lab in range(S.shape[0]):
        S[lab, :] = logpdf_GAU_ND(D, hParams[lab][0], hParams[lab][1])
    return S

# compute log-postorior matrix from log-likelihood matrix and prior array
def compute_logPosterior(S_logLikelihood, v_prior):
    SJoint = S_logLikelihood + vcol(numpy.log(v_prior))
    SMarginal = vrow(scipy.special.logsumexp(SJoint, axis=0))
    SPost = SJoint - SMarginal
    return SPost
                     

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

# Compute log-density for a single sample x (column vector). The result is a 1-D array with 1 element
def logpdf_GAU_ND_singleSample(x, mu, C):
    P = numpy.linalg.inv(C)
    return -0.5*x.shape[0]*numpy.log(numpy.pi*2) - 0.5*numpy.linalg.slogdet(C)[1] - 0.5 * ((x-mu).T @ P @ (x-mu)).ravel()

# Compute log-densities for N samples, arranged as a MxN matrix X (N stacked column vectors). The result is a 1-D array with N elements, corresponding to the N log-densities
def logpdf_GAU_ND_slow(X, mu, C):
    ll = [logpdf_GAU_ND_singleSample(X[:, i:i+1], mu, C) for i in range(X.shape[1])]
    return numpy.array(ll).ravel()

    
# Compute log-densities for N samples, arranged as a MxN matrix X (N stacked column vectors). The result is a 1-D array with N elements, corresponding to the N log-densities
def logpdf_GAU_ND_fast(x, mu, C):
    P = numpy.linalg.inv(C)
    return -0.5*x.shape[0]*numpy.log(numpy.pi*2) - 0.5*numpy.linalg.slogdet(C)[1] - 0.5 * ((x-mu) * (P @ (x-mu))).sum(0)

logpdf_GAU_ND = logpdf_GAU_ND_fast


# Compute and print covariance and correlation matrices
def analyze_covariance_correlation(hParams_MVG):
    for lab in hParams_MVG.keys():
        mu, C = hParams_MVG[lab]
        print(f"Class {lab} - Mean Vector:\n{mu}")
        print(f"Class {lab} - Covariance Matrix:\n{C}")
        Corr = C / (vcol(C.diagonal()**0.5) * vrow(C.diagonal()**0.5))
        print(f"Class {lab} - Correlation Matrix:\n{Corr}\n")


def fit_gaussian_to_features(D, L):
    labelSet = set(L)
    for lab in labelSet:
        DX = D[:, L == lab]
        for i in range(DX.shape[0]):
            feature = DX[i, :]
            mu = feature.mean()
            var = feature.var()
            print(f'Class {lab}, Feature {i+1} - Mean: {mu}, Variance: {var}')
            


def repeat_classification(DTR, LTR, DVAL, LVAL, features):
    # Subset the data to the selected features
    DTR_sub = DTR[features, :]
    DVAL_sub = DVAL[features, :]

    # MVG Model
    hParams_MVG_sub = Gau_MVG_ML_estimates(DTR_sub, LTR)
    S_logLikelihood = compute_log_likelihood_Gau(DVAL_sub, hParams_MVG_sub)
    S_logPost = compute_logPosterior(S_logLikelihood, numpy.ones(2)/2.)
    PVAL = S_logPost.argmax(0)
    error_rate_mvg = (PVAL != LVAL).sum() / float(LVAL.size) * 100
    print(f"MVG Model - Error rate: {error_rate_mvg:.1f}% with features {features+1}")

    # Tied Gaussian Model
    hParams_Tied_sub = Gau_Tied_ML_estimates(DTR_sub, LTR)
    S_logLikelihood = compute_log_likelihood_Gau(DVAL_sub, hParams_Tied_sub)
    S_logPost = compute_logPosterior(S_logLikelihood, numpy.ones(2)/2.)
    PVAL = S_logPost.argmax(0)
    error_rate_tied = (PVAL != LVAL).sum() / float(LVAL.size) * 100
    print(f"Tied Gaussian Model - Error rate: {error_rate_tied:.1f}% with features {features+1}")

    # Naive Bayes Model
    hParams_Naive_sub = Gau_Naive_ML_estimates(DTR_sub, LTR)
    S_logLikelihood = compute_log_likelihood_Gau(DVAL_sub, hParams_Naive_sub)
    S_logPost = compute_logPosterior(S_logLikelihood, numpy.ones(2)/2.)
    PVAL = S_logPost.argmax(0)
    error_rate_naive = (PVAL != LVAL).sum() / float(LVAL.size) * 100
    print(f"Naive Bayes Model - Error rate: {error_rate_naive:.1f}% with features {features+1}")
      


def models(DTR, LTR, DVAL, LVAL):
    hParams_MVG = Gau_MVG_ML_estimates(DTR, LTR)
    
    S_logLikelihood = compute_log_likelihood_Gau(DVAL, hParams_MVG) #hparam contains mu and C
    
    S_logPost = compute_logPosterior(S_logLikelihood, numpy.ones(2)/2.)
       
    # Predict labels
    PVAL = S_logPost.argmax(0)
    print("MVG - Error rate: %.1f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100))    
    
    print()
        
    
    #naive Bayes Guassian Classifier
    
    hParams_Naive = Gau_Naive_ML_estimates(DTR, LTR)
    

    S_logLikelihood = compute_log_likelihood_Gau(DVAL, hParams_Naive)
    S_logPost = compute_logPosterior(S_logLikelihood, numpy.ones(2)/2.)
    PVAL = S_logPost.argmax(0)
    print("Naive Bayes Gaussian - Error rate: %.1f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100))
        
    print()
    #Tied Guassian Classifier

   
    hParams_Tied = Gau_Tied_ML_estimates(DTR, LTR)
    
        
    S_logLikelihood = compute_log_likelihood_Gau(DVAL, hParams_Tied)
    S_logPost = compute_logPosterior(S_logLikelihood, numpy.ones(2)/2.)
    PVAL = S_logPost.argmax(0)
    print("Tied Gaussian - Error rate: %.1f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100))
    
      
            
if __name__ == '__main__':
    
    D, L = Ploting.load('traindata.txt')

    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
   
    
    for lab in [0,1]:
        D = DTR[:, LTR==lab] 
        #print ('MVG - Class', lab)
        mu, C = compute_mu_C(D)
      
        
    hParams_MVG = Gau_MVG_ML_estimates(DTR, LTR)    
    
    models(DTR, LTR, DVAL, LVAL)

    # Analyze Covariance and Correlation Matrices
    analyze_covariance_correlation(hParams_MVG)

    
    print("Gaussian Fit to Each Feature Separately")
    fit_gaussian_to_features(DTR, LTR)
    
    

    # Repeat classification for different feature sets
    print("\nClassification with Features 1-4")
    repeat_classification(DTR, LTR, DVAL, LVAL, numpy.array([0, 1, 2, 3]))
    
    print("\nClassification with Features 1-2")
    repeat_classification(DTR, LTR, DVAL, LVAL, numpy.array([0, 1]))
    
    print("\nClassification with Features 3-4")
    repeat_classification(DTR, LTR, DVAL, LVAL, numpy.array([2, 3]))
    
    
    
    #PCA
    # Apply PCA with different dimensions and classify
    for m in range(1, DTR.shape[0] + 1):
        print(f"\nClassification with PCA (m={m}):")
        P = lab3.compute_pca(DTR, m)
        DTR_PCA = lab3.apply_pca(P, DTR)
        DVAL_PCA = lab3.apply_pca(P, DVAL)
        models(DTR_PCA, LTR, DVAL_PCA, LVAL)

    