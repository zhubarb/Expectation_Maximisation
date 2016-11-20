# http://statweb.stanford.edu/~tibs/stat315a/LECTURES/em.pdf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt; 
from sklearn.preprocessing import StandardScaler
from math import pi
from math import log
from sklearn.mixture import GMM

def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

def get_univ_normal_loglik(data, mean, var):
    '''
    Parameters:
    -------------------------
    data  -
    mean  -
    var   -
    '''
    if len(data.shape) >1: # input data is multidimensional
        raise Exception("This function is valid for univariate data only.")
    else:
        data = pd.Series(data) # make it a Series if it's not so we can sum
        loglik = -0.5*log(pi) - 0.5*log(var) -1/(2*var) *sum( (data-mean)**2 )

    return loglik

def gmm_expectation(data, means, vars, priors):
    '''
    Parameters:
    -------------------------
    data  -
    means - vector fo means for k gaussians
    vars  - vector of vars for k gaussians
    pi    - vector of priors for k gaussians
    Returns the weights (responsibilities) for each observation.
    '''
    k = len(priors) # how many gaussians
    N = len(data) # how many observations

    log_obs = np.zeros(N) # loglik of all individual observations
    lik_obs_i_for_mixt = np.zeros((N,k))
    for i in xrange(N):

        # loop through gaussians to find log_lik for each and then calc weights / responsibilities
        for j in xrange( k ):
            lik_obs_i_for_mixt[i,j] = get_univ_normal_loglik(data.iloc[i], means[j], vars[j])
    weights = np.array(map(lambda x: (np.exp(x) * priors) / (np.exp(x)*priors).sum(),  
                                      lik_obs_i_for_mixt ) 
                       ) #a.k.a. responsibilities 
    
    #log_obs[i] = (lik_obs_i_for_mixt * priors).sum()
    #log_lik = log_obs.sum() # loglik of the entire data as a sum of indiv observations

    return weights

def gmm_loglik(data, means, vars, priors):
    '''
    Parameters:
    -------------------------
    data  -
    means - vector fo means for k gaussians
    vars  - vector of vars for k gaussians
    pi    - vector of priors for k gaussians
    '''
    k = len(priors) # how many gaussians
    N = len(data) # how many observations

    # loop through gaussians to find log_lik for each and then sum
    lik_obs_i_for_mixt = np.zeros(k)
    for k in xrange( k ):
        lik_obs_i_for_mixt[k] = get_univ_normal_loglik(data, means[k], vars[k])

    # loglik is equal to the sum of all k mixture components
    log_lik =(lik_obs_i_for_mixt * priors).sum()

    return log_lik

if __name__ == "__main__": 
    visualise = 0
    old_faithf_dat= "Old_faithful_data.csv"
    data = pd.read_csv(old_faithf_dat)

    data_scaled, scaler= preprocess_data(data)
    data_scaled_df = pd.DataFrame(data_scaled, columns=data.columns.tolist())

    if visualise:
        plt.scatter(data_scaled_df.eruptions, data_scaled_df.waiting)
        plt.show()

    # Univariate # 
    if visualise:
        plt.hist(data.eruptions)
        plt.show()

    # 1. Do Manual EM
    num_iter = 10
    num_mixtures= 2 # how many gaussians
    N = len(data) #  number of observations

    # parameter arrays
    means = np.zeros((num_iter+1,num_mixtures))
    vars = np.zeros((num_iter+1,num_mixtures))
    priors = np.zeros((num_iter+1,num_mixtures))

    # initial values
    means[0]  = np.array([1, 6])
    vars[0]   = np.array([0.2, 0.2])
    priors[0] = np.array([0.4, 0.6])

    for i in xrange(num_iter): # http://statweb.stanford.edu/~tibs/stat315a/LECTURES/em.pdf Slide 13
        
        # calculate expectations (responsibilities)
        expectations = gmm_expectation(data.eruptions, means[i], vars[i], priors[i])

        # maximise likelihood for all three params given the expectations 
        means[i+1]  = [( (expectations[:,j] * data.eruptions).sum()) / expectations[:,j].sum()  
                       for j in xrange(num_mixtures)]
        vars[i+1]   = [ (expectations[:,j]*(data.eruptions - means[i+1,j])**2).sum() /  expectations[:,j].sum()
                       for j in xrange(num_mixtures)]
        priors[i+1] = [ expectations[:,j].sum() / N   for j in xrange(num_mixtures)]

        print str(i) + ' |means :' + str(means[i+1].round(2)) + \
                       ' |vars :' + str(vars[i+1].round(2)) +  \
                        '|priors :' + str(priors[i+1].round(2))


    # 1. Do ScikitLearn GMM
    gmm = GMM(n_components=num_mixtures)
    gmm.fit(data.eruptions)
    print 'Sklearn finds priors: '  + str( np.round(gmm.weights_, 2) )
    print 'Sklearn finds means:  '  + str( np.round(gmm.means_, 2  ) )
    print 'Sklearn finds vars: '    + str( np.round(gmm.covars_, 2 ) )

