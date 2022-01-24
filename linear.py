# Posterior distribution and evidence in the GLM method
#Parameters 
# data set as, (x_obs_y_obs,y_obs_err), length of all vector should be the same
# 
#
#
#


import numpy as np


class linear_reg:

    def __init__(self,x_obs,y_obs,y_obs_err,n_par,fun_list):
        self.X = x_obs
        self.Y = y_obs
        self.err = y_obs_err
        self.n_par = n_par
        self.funcs = fun_list

        
        if len(x_obs)!= len(y_obs):
            raise Exception("Length of X and Y is not the same")

        if len(y_obs)!= len(y_obs_err):
            raise Exception("Length of Y and Y_err is not the same")

        if len(fun_list)!= n_par:
            raise Exception("Number of functions is not the same as number of parameters")   


        self.n_obs = len(x_obs)
        self.F = np.zeros(shape=(self.n_obs,self.n_par))
        self.A = np.zeros(shape=(self.n_obs,self.n_par))

        for j in range(self.n_par):
            self.F[:,j] = [self.funcs[j](x) for x in self.X]
            self.A[:,j] = self.F[:,j]/self.err

        self.b = self.Y/self.err
        self.L = np.dot(self.A.T,self.A)
        self.L_inv = np.linalg.inv(self.L)
        self.theta_0 = self.L_inv@self.A.T@self.b        


    def max_likelihood(self):

        return self.theta_0


    def post_dist(self,pri_mean,pri_cov):

        self.prior_mean = pri_mean
        self.prior_cov = pri_cov
        self.theta_0 = self.L_inv@self.A.T@self.b

        self.pos_cov = self.L + self.prior_cov
        self.pos_cov_inv = np.linalg.inv(self.pos_cov)
        t = self.L@self.theta_0 + self.prior_cov@self.prior_mean
        self.pos_mean = self.pos_cov_inv@t
    
        return self.pos_mean,self.pos_cov_inv    



    def evidenc(self,pri_mean,pri_cov):


        self.theta_0 = self.L_inv@self.A.T@self.b
        pow_l0 = -0.5*np.dot((self.b-np.dot(self.A,self.theta_0)).T,self.b-np.dot(self.A,self.theta_0))
        t1 = pow(2*np.pi,self.n_obs/2)*np.prod(self.err)
        #L0 = (1./t1)*np.exp(pow_l0)

        self.pos_cov = self.L + pri_cov
        self.pos_cov_inv = np.linalg.inv(self.pos_cov)
        t2 = pow(np.linalg.det(self.pos_cov)/np.linalg.det(pri_cov),-0.5)
        D1 = 0.5*(  self.theta_0.T@self.L@self.theta_0   +  pri_mean.T@pri_cov@pri_mean  )
        D2 = 0.5*( (self.theta_0.T@self.L + pri_mean.T@pri_cov) @self.pos_cov_inv @ (self.L@self.theta_0 + pri_cov@pri_mean)     ) 
        #t3 = np.exp(D2-D1)
        #evid = L0*t2*t3

        return np.log(t2) - np.log(t1) + pow_l0 + D2 - D1    
