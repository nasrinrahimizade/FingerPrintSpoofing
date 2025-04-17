
import numpy
import matplotlib.pyplot as plt
import Ploting

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

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

logpdf_GAU_ND = logpdf_GAU_ND_slow

def compute_ll(X, mu, C):
    return logpdf_GAU_ND(X, mu, C).sum()

# Compute ML estimates for each class and feature
def compute_ml_estimates(D, L):
    ml_estimates = {}
    for cls in numpy.unique(L):
        D_cls = D[:, L == cls]
        mu, C = compute_mu_C(D_cls)
        ml_estimates[cls] = {'mu': mu, 'C': C}
    return ml_estimates


# Plot distribution density on top of normalized histogram
def plot_density_histogram(D, L, ml_estimates):
    hFea = {
        0: 'f1',
        1: 'f2',
        2: 'f3',
        3: 'f4',
        4: 'f5',
        5: 'f6'
    }
    
    for cls in numpy.unique(L):
        for feat_idx in range(D.shape[0]):
            plt.figure()
            plt.xlabel(hFea[feat_idx])
            
            # Plot histogram
            D_cls_feat = D[feat_idx, L == cls]
            plt.hist(D_cls_feat.ravel(), bins=10, density=True, alpha=0.4, label='Class {}'.format(cls))
            
            # Compute ML estimate for the current class and feature
            mu = ml_estimates[cls]['mu']
            C = ml_estimates[cls]['C']
            XPlot = numpy.linspace(D_cls_feat.min(), D_cls_feat.max(), 1000)
            log_density = logpdf_GAU_ND(vrow(XPlot), mu, C)
            plt.plot(XPlot.ravel(), numpy.exp(log_density), label='Density', color='red')
            
            plt.legend()
            plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
            plt.savefig('hist_cls{}_feat{}.png'.format(cls, feat_idx))
    plt.show()
    
    
    
if __name__ == '__main__':

    # Load dataset and labels
    D, L = Ploting.load('traindata.txt')  # Replace 'your_dataset.txt' with the path to your dataset file
    
    # Compute ML estimates for each class and feature
    ml_estimates = compute_ml_estimates(D, L)
    
    # Plot distribution density on top of normalized histogram
    plot_density_histogram(D, L, ml_estimates)