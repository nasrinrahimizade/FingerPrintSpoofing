
import pca
import lda
import Ploting
import numpy
import matplotlib.pyplot as plt

import lda 

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


if __name__ == '__main__':

    D, L = Ploting.load('traindata.txt')
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    # Solution without PCA pre-processing and threshold selection. The threshold is chosen half-way between the two classes
    print('Solution without PCA pre-processing')
    ULDA = lda.compute_lda_JointDiag(DTR, LTR, m=1)

    DTR_lda = lda.apply_lda(ULDA, DTR)

    if DTR_lda[0, LTR==0].mean() > DTR_lda[0, LTR==1].mean():
        ULDA = -ULDA
        DTR_lda = lda.apply_lda(ULDA, DTR)

    DVAL_lda  = lda.apply_lda(ULDA, DVAL)

    threshold = (DTR_lda[0, LTR==0].mean() + DTR_lda[0, LTR==1].mean()) / 2.0 # Estimated only on model training data
    print ('threshold:',threshold)
    PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
    PVAL[DVAL_lda[0] >= threshold] = 1
    PVAL[DVAL_lda[0] < threshold] = 0
    print('Labels:     ', LVAL)
    print('Predictions:', PVAL)
    print('Number of erros:', (PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
    print('Error rate: %.1f%%' % ( (PVAL != LVAL).sum() / float(LVAL.size) *100 ))
          
    print('different thresholds:')
    # Initial threshold based on class means
    initial_threshold = (DTR_lda[0, LTR == 0].mean() + DTR_lda[0, LTR == 1].mean()) / 2.0

    # Define a range of thresholds to test
    threshold_range = numpy.linspace(initial_threshold - 1.0, initial_threshold + 1.0, num=20)

    best_threshold = initial_threshold
    best_error_rate = float('inf')

    for threshold in threshold_range:
        PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
        PVAL[DVAL_lda[0] >= threshold] = 1
        PVAL[DVAL_lda[0] < threshold] = 0
        error_rate = (PVAL != LVAL).sum() / float(LVAL.size) * 100
        print(f'Threshold: {threshold:.2f}, Error rate: {error_rate:.1f}%')
        
        if error_rate < best_error_rate:
            best_error_rate = error_rate
            best_threshold = threshold

    print(f'Best threshold: {best_threshold:.2f}, Best error rate: {best_error_rate:.1f}%')
   
    
   
    
   
    # Initialize variables for recording results
    error_rates = []
    m_values = [1, 2, 3, 4, 5, 6]  # Example values of m to test
    
    for m in m_values:
        print(f"PCA with dimension {m}")
        
        # Apply PCA on training and validation data
        UPCA = pca.compute_pca(DTR, m=m)
        DTR_pca = pca.apply_pca(UPCA, DTR)
        DVAL_pca = pca.apply_pca(UPCA, DVAL)
        
        # Apply LDA on PCA-transformed training data
        ULDA = lda.compute_lda_JointDiag(DTR_pca, LTR, m=1)
        DTR_lda = lda.apply_lda(ULDA, DTR_pca)
        
        # Check the orientation of the LDA direction
        if DTR_lda[0, LTR==0].mean() > DTR_lda[0, LTR==1].mean():
            ULDA = -ULDA
            DTR_lda = lda.apply_lda(ULDA, DTR_pca)
        
        # Apply LDA on PCA-transformed validation data
        DVAL_lda = lda.apply_lda(ULDA, DVAL_pca)
        
        # Calculate threshold
        threshold = (DTR_lda[0, LTR==0].mean() + DTR_lda[0, LTR==1].mean()) / 2.0
        
        # Make predictions based on threshold
        PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
        PVAL[DVAL_lda[0] >= threshold] = 1
        PVAL[DVAL_lda[0] < threshold] = 0
        
        # Compute error rate
        error_rate = (PVAL != LVAL).sum() / float(LVAL.size) * 100
        error_rates.append(error_rate)
        
        # Print results
        print(f"m = {m}, Error rate: {error_rate:.1f}%")
    
    # Plotting the error rates versus m
    plt.figure()
    plt.plot(m_values, error_rates, marker='o')
    plt.xlabel('Number of PCA dimensions (m)')
    plt.ylabel('Error rate (%)')
    plt.title('Error rate vs. Number of PCA dimensions')
    plt.grid(True)
    plt.xticks(m_values)
    plt.show()
   
    
   
   