import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import gamma 

from scipy.stats import gamma as gammadist



# Number of samples drawn
sample_number=100000

# Functions that we are using

def poly_want_to_sample(x):
    """The polynomial we want to sample - known up to the normalising constant """
    return 0.3 * np.exp(-0.2 * x * x) + 0.7 * np.exp(-0.2 * (x - 10) * (x - 10))

def posterior(lambda_val, alpha, beta, data):
    """function that gives the values of the gamma posterior for different values of alpha, beta and data"""

    count_no = len(data)
    sum_val = sum(data)
    new_beta = beta + count_no
    new_alpha = alpha + sum_val

    numerator = (
        (new_beta ** new_alpha)
        * np.exp(-lambda_val * new_beta)
        * (lambda_val ** (new_alpha - 1))
    )
    denominator = gamma(new_alpha)

    val = numerator / denominator

    return val

data_var=list(np.random.poisson(6, 2))

def mcmc_threshold(x, y):
    """Function that generates the mcmc probaility threshold for accepting the proposed state
         x is the current state
         y is the proposed new state
         norm.pdf(x,y,100) is the function x of a normal distribution with mean y and variance 100
         """
    ratio = (posterior(y,0.3,0.1, data_var) * norm.pdf(x, y, 100)) / (
       posterior(x,0.3,0.1, data_var) * norm.pdf(y, x, 100)
    )

    return   min(ratio, 1)

# Create a data frame to hold the results
data_store = pd.DataFrame(index=range(sample_number), columns=["state_of_chain"])


# Create an initial value to start the chain
initial_value=np.random.normal(10,10)

state = initial_value

# Run the samples
for i in range(0, sample_number):
    # Generate the proposed state
    prop_value = np.random.normal(state, 100)

    # Generate a number between 1 and 0 to simulate probability of acceptance
    prob_sample = np.random.uniform(0, 1)

    # Condition for accepting the proposed state 
    # the probability of the state being accepted is prob_sample
    if prob_sample < mcmc_threshold(state, prop_value):
        state = prop_value
    else:
        state = state
    # Storing the value in the dataframe
    data_store["state_of_chain"].loc[i] = state
    # Prints charts at regular intervals
    if i % 10000 == 0:
        plt.figure(figsize=(14, 10))
        plt.xlim(0, 15)
        plt.hist(data_store["state_of_chain"], density=True, bins=50)
        plt.title("Distribution of mcmc chain's " + str(i) + "'th run", weight="bold")
        x = np.arange(0, 15, 0.1)
        # Plots the probability distribution we want to sample scaled to chart
        plt.plot(x, posterior(x,0.3,0.1,data_var)*0.2, color="red")

data_mean=data_store["state_of_chain"].mean()
data_var=data_store["state_of_chain"].var()

alpha_est=(data_mean**2)/(data_var**2)
beta_est=(data_mean)/(data_var**2)

#gamma.pdf(x,alpha_est,loc=0,scale=(1/beta_est))

plt.figure(figsize=(14, 10))
plt.xlim(0, 15)
plt.hist(data_store["state_of_chain"], density=True, bins=50)
plt.title("Distribution of mcmc chain's " + str(i) + "'th run", weight="bold")
x = np.arange(0, 15, 0.1)
 # Plots the probability distribution we want to sample scaled to chart
#plt.plot(x, posterior(x,0.3,0.1,data_var)*0.2, color="red")
plt.plot(x, gammadist.pdf(x,alpha_est,loc=0,scale=(1/beta_est)), color="blue")
print(plt.show())





print(plt.show()) 
plt.savefig("figurename.png")
plt.close()
