import numpy as np
import math
import matplotlib.pyplot as plt

#### 11 July 13: E-M Coin Toss Example as given in Stanford EM tutorial* #### 
# * http://algorithmicalley.com/archive/2013/03/29/the-expectation-maximization-algorithm.aspx

def get_mn_log_likelihood(obs,probs):
    """ Return the (log)likelihood of obs, given the probs"""
    # Multinomial Distribution Log PDF
    # ln (pdf)      =             multinomial coeff            *   product of probabilities
    # ln[f(x|n, p)] = [ln(n!) - (ln(x1!)+ln(x2!)+...+ln(xk!))] + [x1*ln(p1)+x2*ln(p2)+...+xk*ln(pk)]     

    multinomial_coeff_denom= 0
    prod_probs = 0
    for x in range(0,len(obs)): # loop through state counts in each observation
        multinomial_coeff_denom = multinomial_coeff_denom + math.log(math.factorial(obs[x]))
        prod_probs = prod_probs + obs[x]*math.log(probs[x])

    multinomial_coeff = math.log(math.factorial(sum(obs))) -  multinomial_coeff_denom
    likelihood = multinomial_coeff + prod_probs
    
    return likelihood

# The real situation for the 5 experiments which individually comprise 10 tosses with the corresponding coins (A or B)

# 1st:  Coin B, {HTTTHHTHTH}, 5H,5T
# 2nd:  Coin A, {HHHHTHHHHH}, 9H,1T
# 3rd:  Coin A, {HTHHHHHTHH}, 8H,2T
# 4th:  Coin B, {HTHTTTHHTT}, 4H,6T
# 5th:  Coin A, {THHHTHHHTH}, 7H,3T
# so, from MLE: pA(heads) = 0.80 and pB(heads)=0.45

# represent the experiments
head_counts = np.array([5,9,8,4,7])
tail_counts = 10-head_counts
experiments = zip(head_counts,tail_counts)

# initialise the pA(heads) and pB(heads)
pA_heads = np.zeros(100); pA_heads[0] = 0.60
pB_heads = np.zeros(100); pB_heads[0] = 0.50

# E-M begins!
delta = 0.001  
j = 0 # iteration counter
improvement = float('inf')
while (improvement>delta):

    # expactations with coins A and B with the initial pA_heads and pB_heads parameters
    expectation_A = np.zeros((5,2), dtype=float) 
    expectation_B = np.zeros((5,2), dtype=float)

    for i in range(0,len(experiments)):

        e = experiments[i] # i'th experiment

        ll_A = get_mn_log_likelihood(e,np.array([pA_heads[j],1-pA_heads[j]])) # loglikelihood of e given coin A
        ll_B = get_mn_log_likelihood(e,np.array([pB_heads[j],1-pB_heads[j]])) # loglikelihood of e given coin B
        
        weightA = math.exp(ll_A) / ( math.exp(ll_A) + math.exp(ll_B) ) # corresponding weight of A proportional to likelihood of A
        weightB = math.exp(ll_B) / ( math.exp(ll_A) + math.exp(ll_B) ) # corresponding weight of B proportional to likelihood of B                            
                                     
        # Find Expectations of coins A and B for the i'th experiment
        expectation_A[i] = np.dot(weightA, e) # weightA * e does not work.
        expectation_B[i] = np.dot(weightB, e)

    # Find the parameters that maximise the accumulated (sum) expectations of A and B 
    # (through counting, i.e. MLE of pA_heads and pB_heads for multinomial distributions)
    pA_heads[j+1] = sum(expectation_A)[0] / sum(sum(expectation_A)); 
    pB_heads[j+1] = sum(expectation_B)[0] / sum(sum(expectation_B)); 

    improvement = max( abs(np.array([pA_heads[j+1],pB_heads[j+1]]) - np.array([pA_heads[j],pB_heads[j]]) ))
    j = j+1


plt.figure();
plt.plot(range(0,j),pA_heads[0:j], 'r--')
plt.plot(range(0,j),pB_heads[0:j])
plt.show()

