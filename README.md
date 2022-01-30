# GLM 
# Gaussian linear model 

## The Gaussian linear model provides a unique approach to obtain the posterior distribution as well as the Bayesian evidence analytically.

## Arguments:

Database should be given as (x_obs,y_obs,y_obs_err) 
n_par: number of free parameres
fun_list: list of callable functions. The number of functions should be same as the number of parameters

## Methods:

max_likelihood(): Return the best value of parameters to maximize the likeliood function.

post_dist(pri_mean,pri_cov): given prior information, the method gives the posterior mean and covariance matrix 
                              pri_mean: mean of prior on the parameter.
                              pri_cov: covariance matrix of prior on the parameter.

evidenc(pri_mean,pri_cov): given prior information, the method gives the ln(evidenc)
                              pri_mean: mean of prior on the parameter.
                              pri_cov: covariance matrix of prior on the parameter.
