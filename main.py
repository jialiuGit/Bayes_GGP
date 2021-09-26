#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 19:51:45 2021

@author: jialiu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 15:22:39 2021

@author: jialiu
 
Bayesian inference---Gibbs-Metropolise sampling
paras
mu---field mean
Sig -- field covariance 
augmented variable of length --r     
    
"""
 
import numpy as np 
from  PN_Bayes import *
from  PNcov import *
import pickle
import time   
import argparse
import pdb
#run python
#python3 main.py  large 8 20000 5000 --Initials_xi 30 3 0.5 6 15 1.0 --Initials_g -18 0 0  --directory results --rstname tk03_test_large

#---data_name ----name of user's data, we need data--- unit (spherical) directional data in cartesian coordinate, and lat --- latitude to run the code
#---degree------- degree in SH, l, we use a (default) value is 8, here is an input 
#---nMCMC ------- number of MCMC runs
#---burnin--------number of burnin in MCMC
#---Intials_xi----initial values sig10_2, sig11_2,sig20_2,sig21_2, alpha, beta 
#---Intials_g---- initial values of means of g_1^0 , g_2^0, g_3^0,...,g_l^0
#---mu_ast--------parameter mu star in the Mirror random walk proposal
#---beta_model----if beta is treated as a fixed value, |xi| = 5, otherwise |xi| - 6
#---fileformat----file format when saving the results, default is '.pkl'
#---GGPmodel-----the exsiting GGPmodel, default is tk03
#---directory----directory where is to save the results
#---rstname------name of the results
#---method ------ we provide three different solutions to update field variation, 'ab_joint'---update xi where alpha and beta are together; 
#'ab_ind' --update xi where alpha and beta is independently, 'x2' ---update xi2 . 


parser = argparse.ArgumentParser(description='main script to run on server')
parser.add_argument('data_name', metavar='data_name', nargs='+',help='name of data') #
parser.add_argument('degree', metavar='degree', type=int, nargs='+',help='degree in SH')
parser.add_argument('nMCMC', metavar='nMCMC', type=int, nargs='+',help='number of MCMC iterations')
parser.add_argument('burnin', metavar='burnin', type=int, nargs='+',help='number of burnin in MCMC')
parser.add_argument('--Initials_xi', metavar='Initials_xi', type=float, nargs='+',help='initial values of variation parameter') 
parser.add_argument('--Initials_g', metavar='Initials_g', type=float, nargs=3, help='initial values of field parameter') 
parser.add_argument('--mu_ast', metavar='mu_ast', type=float, nargs='+',help='mu of RW mirror proposal')
parser.add_argument('--beta_model', metavar='beta_model', nargs='+',help='model of beta (fixed/r.v.)')
parser.add_argument('--fileformat', metavar='fileformat', nargs='+',help='figure fileformat (pkl)')
parser.add_argument('--GGPmodel', metavar='GGPmodel', nargs='+',help='name of GGP model')
parser.add_argument('--directory', metavar='directory', nargs='+',help='Subdirectory for output-files')
parser.add_argument('--rstname', metavar='rstname', nargs='+',help='result name in pkl-file')
parser.add_argument('--method', metavar='method', nargs='+',help='simulation method') #

args = parser.parse_args()

data_name =str(args.data_name[0])
degree = np.int(args.degree[0])
nMCMC = np.int(args.nMCMC[0])
burnin = np.int(args.burnin[0])
Initials_xi =np.float32(args.Initials_xi)
sig10_2, sig11_2, sig20_2, sig21_2, alpha, beta = Initials_xi
sig22_2 = sig20_2

Initials_g =np.float32(args.Initials_g)
G = Initials_g

if args.data_name is None:
    data_name = 'large'    
else:
	data_name=str(args.data_name[0])
        
if args.beta_model is None:
	beta_model="fixed"
else:
	beta_model=str(args.beta_model[0])

if args.mu_ast is None:  #run the code get mu_ast from burnin or some good choices such as tk03
    if beta_model == 'fixed': 
        mu_ast = 2*np.array([6.4, 1.7, 0.6, 2.2, 7.5])
    else:
        mu_ast = 2*np.array([6.4, 1.7, 0.6, 2.2, 7.5, 7.5*3.8])
else:
	mu_ast=2*np.float32(args.mu_ast) 
#pdb.set_trace()
if args.fileformat is None:
	fileformat="pkl"
else:
	fileformat=str(args.fileformat[0])

if args.GGPmodel is None:
	GGPmodel="tk03"
else:
	GGPmodel=str(args.GGPmodel[0])
    
if args.method is None:
	method="ab_joint"
else:
	method=str(args.method[0])    

 #load your data: we need 'data = directional data' and 'lat = latitude' to run the code!  
if data_name == 'large':
     with open('simu_large_data.pkl',  'rb') as f:  
        tk03_data, tk03_real_inten, tk03_lat, CJ98_data, CJ98_real_inten, CJ98_lat = pickle.load(f)
     if GGPmodel == 'tk03':
        data = tk03_data           
     elif GGPmodel == 'CJ98':
        data = CJ98_data
     lat = tk03_lat 
#pdb.set_trace()  
    
if beta_model == 'fixed':
    cof_K = cof_K1(degree, lat, beta)
else:
    cof_K = cof_K2(degree, lat)

    
directory		 = str(args.directory[0])
rstname		   = str(args.rstname[0])
  

def update_Sigma_AM_onestp(Y,  mu,  lat, degree,  cof_K, sig10_2, sig11_2,sig20_2,sig21_2, alpha,beta, Sigma0, sd, mu_ast):
 
    #update alpha and beta together, alpha * beta is random variables #default method
    uvec = Y - mu    
    cov_len  =cof_K.shape[0]
     
    if cov_len  ==5:
         x1 = np.hstack((np.array(np.sqrt([sig10_2, sig11_2,sig20_2,sig21_2]) , dtype=float) ,alpha))         
         x0 = np.hstack((np.array(np.sqrt([sig10_2, sig11_2,sig20_2,sig21_2]), dtype=float) ,alpha))         
    else:
        x1 = np.hstack((np.array(np.sqrt([sig10_2, sig11_2,sig20_2,sig21_2]), dtype=float) ,alpha, alpha*beta))       
        x0 = np.hstack((np.array(np.sqrt([sig10_2, sig11_2,sig20_2,sig21_2]), dtype=float) ,alpha, alpha*beta))       
 
    for j in range(cov_len):   
        sampXI = list(x0)  #will not change x0  
        go_on = 1
        while go_on:
            xi_j = np.random.normal(mu_ast[j] - x1[j], sd[j])         
            if xi_j > 0.0:  #positive constraints
               go_on = 0
        sampXI[j] = xi_j  
        #print(xi_j,j)
        #pdb.set_trace()
       
        logMHrat  = log_lik_diff(uvec, cof_K, np.array(sampXI)**2, x0**2 )                
        MHrat = np.exp(logMHrat)          
        vrat = np.random.uniform(0,1)          
        if  (MHrat >  vrat):
            x0[j] = xi_j              
    if cov_len == 5:
        dist = np.abs(np.sum(x0 -  x1)) 
    else:
        #pdb.set_trace()
        dist = np.abs(np.sum(x0[0:-1] -  x1[0:-1])) 
    if (dist) > 1e-5:   
        if cov_len  ==6:
              sig10_2, sig11_2, sig20_2, sig21_2 =x0[0:4]**2           
              beta =  x0[-1]/x0[-2]
        else:
            sig10_2, sig11_2, sig20_2, sig21_2 =  x0[0:4]**2  
        alpha = x0[4]
        sig22_2 = sig20_2       
        loglik = log_post_fun(uvec, cof_K, x0**2)        
        lat0 = 0
        Sigma = np.zeros((n,3,3))
        for i in range(len(Y)):
            if lat[i] != lat0:  
                Sigma_i = psv.Cov(alpha,beta,lat[i], degree,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2)                  
                Sigma[i] = Sigma_i            
            else:
                Sigma[i] = Sigma[i-1] 
            lat0 = lat[i] 
    else:
        sig22_2 = sig20_2  
        Sigma = Sigma0 
        loglik = log_post_fun(uvec, cof_K, x1**2)   
      
    return sig10_2, sig11_2, sig20_2, sig21_2,sig22_2, alpha,beta, Sigma, loglik 

 
def update_Sigma_AM_onestp_beta(Y,  mu,  lat, degree,  cof_K, sig10_2, sig11_2,sig20_2,sig21_2, alpha,beta, Sigma0, sd, mu_ast):
    #update alpha and beta separately, beta is random variables
    sig2 = 100 #---prior
    uvec = Y - mu
    
    
    cov_len  =cof_K.shape[0]
     
    if cov_len  ==5:
         x1 = np.hstack((np.array(np.sqrt([sig10_2, sig11_2,sig20_2,sig21_2]) , dtype=float) ,alpha))         
         x0 = np.hstack((np.array(np.sqrt([sig10_2, sig11_2,sig20_2,sig21_2]), dtype=float) ,alpha))         
         
    else:
        x1 = np.hstack((np.array(np.sqrt([sig10_2, sig11_2,sig20_2,sig21_2]), dtype=float) ,alpha, beta))     
        x0 = np.hstack((np.array(np.sqrt([sig10_2, sig11_2,sig20_2,sig21_2]), dtype=float) ,alpha, beta))        
 
    for j in range(cov_len):   
        sampXI = list(x0)  #will not change x0  
        go_on = 1
        while go_on: 
            xi_j = np.random.normal(mu_ast[j] - x1[j], sd[j])  
            if xi_j > 0.0: 
                go_on = 0
           
        sampXI[j] = xi_j  
      
        if j == 5:
             xi_new = np.hstack((x0[0:5]**2, (x0[4]*xi_j)**2 ))
             xi_old = np.hstack((x0[0:5]**2, (x0[4]*x0[j])**2 ))
            # pdb.set_trace()
        else:
             xi_new = np.array(sampXI)**2
             xi_old = x0**2             
        
        logMHrat  = log_lik_diff(uvec, cof_K, xi_new , xi_old) +  x0[j]**2/sig2 -  xi_j**2/sig2         
        #pdb.set_trace()       
        MHrat = np.exp(logMHrat)          
        vrat = np.random.uniform(0,1)          
        if  (MHrat >=  vrat):
            x0[j] = xi_j        
    
    dist = np.abs(np.sum(x0 -  x1))    
    if (dist) > 1e-6: 
        alpha = x0[4]
        if cov_len  ==6:
              sig10_2, sig11_2, sig20_2, sig21_2 =x0[0:4]**2   
              #alpha = x0[4]
              beta =  x0[-1]
        else:
            sig10_2, sig11_2, sig20_2, sig21_2 =x0[0:4]**2         
        sig22_2 = sig20_2
        #alpha = np.sqrt(alpha2) 
        if cov_len == 5:
            loglik = log_post_fun(uvec, cof_K, x0**2 ) 
        else:
            loglik = log_post_fun(uvec, cof_K,  np.hstack((x0[0:5]**2, (x0[4]*x0[-1])**2 ))) 
        lat0 = 0
        Sigma = np.zeros((n,3,3))
        for i in range(len(Y)):
            if lat[i] != lat0:  
                Sigma_i = psv.Cov(alpha,beta,lat[i], degree,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2)                  
                Sigma[i] = Sigma_i            
            else:
                Sigma[i] = Sigma[i-1] 
            lat0 = lat[i] 
    else:
        sig22_2 = sig20_2  
        Sigma = Sigma0 
        if cov_len == 5:
             loglik = log_post_fun(uvec, cof_K, x1**2) 
        else:
            loglik = log_post_fun(uvec, cof_K,  np.hstack((x1[0:5]**2, (x1[4]*x1[-1])**2 ))) 
      
    return sig10_2, sig11_2, sig20_2, sig21_2,sig22_2, alpha,beta, Sigma, loglik 


def update_Sigma_AM_onestp2(Y,  mu,  lat, degree,  cof_K, sig10_2, sig11_2,sig20_2,sig21_2, alpha,beta, Sigma0, sd, mus_ast):
     
    #update xi2---- sig10_2, sig11_2,sig20_2,sig21_2, alpha**2,  (alpha*beta)**2 
    def rep_array(XI,x_j,j):
        XI[j] = x_j
        return XI
    
    
    def strwhat(x0,sig):
        tmp = np.random.uniform(0,1,3)
        u1, u2, u3 = tmp
        a = 1
        b = 1.35
        term1 = a/(3*b - 2*a)
        if u1 < term1:
            y = a*np.power(u2,1/3)
        else:
            y = np.random.uniform(a,b,1)[0]
        if u3 < 0.5:
            y = - y
        xprime = x0 + sig*y        
        return xprime 
  
    uvec = Y - mu
    
    
    cov_len  =cof_K.shape[0]     
    if cov_len  ==5:
         x0 = np.array([sig10_2, sig11_2, sig20_2, sig21_2,alpha**2], dtype=float) 
         x1 = np.array([sig10_2, sig11_2, sig20_2, sig21_2,alpha**2], dtype=float)
         mu_ast = 2*np.array([40.96,  2.89,  0.36,  4.84, 56.25])  #xi2 or other choices from burnin
         
    else:       
        x0 = np.array([sig10_2, sig11_2, sig20_2, sig21_2,alpha**2,  (alpha*beta)**2], dtype=float)
        x1 = np.array([sig10_2, sig11_2, sig20_2, sig21_2,alpha**2,  (alpha*beta)**2], dtype=float)
        mu_ast = 2*np.array([40.96,  2.89,  0.36,  4.84, 56.25,56.25*14.4])       
    
    for j in range(cov_len):   
        sampXI = list(x0)  #will not change x1  
        go_on = 1
        while go_on:   
            xi_j = np.random.normal(mu_ast[j] - x1[j], sd[j])  
            if xi_j > 0.0:  
                go_on = 0
        sampXI[j] = xi_j  
        #print(xi_j,j)
        #print(xi_j)
        logMHrat  = log_lik_diff(uvec, cof_K, sampXI, x0 )    
         
        MHrat = np.exp(logMHrat) 
         
        vrat = np.random.uniform(0,1)          
        if  (MHrat >  vrat):
            x0[j] = xi_j     
    if (np.abs(np.sum(x0[0:cov_len-1] -  x1[0:cov_len-1]))  ) > 1e-5: 
        if cov_len  ==6:
              sig10_2, sig11_2, sig20_2, sig21_2,alpha2, tau2 =x0
              beta = np.sqrt(tau2/alpha2)
        else:
            sig10_2, sig11_2, sig20_2, sig21_2, alpha2 =x0     
        sig22_2 = sig20_2
        alpha = np.sqrt(alpha2) 
        
        loglik =  log_post_fun(uvec, cof_K, x0)        
        lat0 = 0
        Sigma = np.zeros((n,3,3))
        for i in range(len(Y)):
            if lat[i] != lat0:  
                Sigma_i = psv.Cov(alpha,beta,lat[i], degree,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2)                  
                Sigma[i] = Sigma_i            
            else:
                Sigma[i] = Sigma[i-1] 
            lat0 = lat[i] 
    else:
        sig22_2 = sig20_2  
        Sigma = Sigma0 
        loglik = log_post_fun(uvec, cof_K, x1)  
    
    return sig10_2, sig11_2, sig20_2, sig21_2,sig22_2, alpha,beta, Sigma, loglik 

 
def log_post_diff(u, cof_K, sig_paras, x0, a, b):

    n = len(u)
    cov_len = cof_K.shape[0]
 
    diffloglik = 0
#    
    def log_inverse_gamma(x, a,b):
        log_ig = 0
        for j in range(len(x)):
            log_ig = log_ig - ((a[j] + 1)*np.log(x[j]) + b[j]/x[j])
        return log_ig
#    
    def log_normal(x, mu,sig2):
        log_norm = 0         
        for j in range(len(x)):
            log_norm = log_norm -  (x[j]-mu[i])**2/sig2
        return log_norm      
    for i in range(n):    #This step can be done in parallel!     
        K = 0
        K1 = 0
        for j in range(cov_len): #covariates j                          
            K = K + sig_paras[j]* cof_K[j][i]
            K1 = K1 +   x0[j]* cof_K[j][i]
            
            Kpre = Omega_mat(K)
            Kpre1 = Omega_mat(K1)
       
        A = u[i].reshape(3,1)*u[i].reshape(1,3)     
        diffloglik = diffloglik + np.matrix.trace( np.dot(A, Kpre1) )   -   np.matrix.trace( np.dot(A, Kpre) )   +  np.log(np.linalg.det(K1) ) -  np.log(np.linalg.det(K) ) 
   
    prior =   log_normal(np.sqrt(sig_paras), a,b) - log_normal(np.sqrt(x0), a,b)
    post = 0.5*diffloglik + prior
    return post

def same_sign(x,y):
    #pdb.set_trace()
    if abs(x)<= 1e-3 or abs(y) <= 1e-3:
        sign = 1
    elif abs(x) + abs(y) == abs(x + y):
        sign = 1
    else:
        sign = 0
    return sign
        
        
#----------------------main body------------------------
n = len(data)
cov_len = cof_K.shape[0] 
R = np.ones(n) 
G4 =0.0
 
start = time.time()
log_fullpost = np.zeros((nMCMC+1,1))
logpost_g = np.zeros((nMCMC+1,1))
Sig_y = np.zeros((nMCMC+1,1))


C0 = np.diag(np.ones(cov_len))
J = len(G) 
sum_G = np.zeros((1,J))
sum_sig_l_02 = np.zeros((1,7)) 
t = 0
 
mu = np.zeros((n,3))
sig_l_02_samp = np.zeros((nMCMC+1,7))
G_samp =  np.zeros((nMCMC+1,len(G)))

   
seqJ =  np.array(range(J))

#design matrix from data---only compute once
X = np.zeros((n,3,len(G)))
lat0 =0

Sigma = np.zeros((n,3,3))
#X1 is a design matrix
for i in range(n):   
    if lat[i] != lat0:
         X1  = deg_mat_zonal(lat[i],G4,a_r=1.0) #ith design matrix  
         Xtmp= X1
         Sigma_i = psv.Cov(alpha,beta,lat[i], degree,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2) 
         Sigma[i] = Sigma_i
    else:
         X1  = Xtmp
         Sigma[i] =  Sigma[i-1] 
         #pdb.set_trace()
    #print(Sigma[i] , Sigma[i-1] )
    X1 = np.vstack((X1[0],[0,0,0],X1[1]))     
    X[i] = X1
    lat0 = lat[i]
 
while t <= nMCMC:
    #update length paras R----given G compute mu 
    Y = np.zeros((n,3))   
    lat0 = 0
    for i in range(n):            
        lat_i =  lat[i]
        #print(lat_i, lat0)
        if lat_i != lat0:       
            mu[i] = np.array([np.sum(G*X[i,0]),0.0, np.sum(G*X[i,2])])            
        else:
            mu[i] = mu[i-1]            
        lat0 = lat_i       
        R[i] = slice_samp_r_weight(R[i], data[i],mu[i],Sigma[i],3)      
        Y[i] = R[i]*data[i]   
 #-------------------------------------------------------------------------------------        
    if cov_len == 5:
        sig_paras = np.array([sig10_2, sig11_2, sig20_2, sig21_2, alpha])        
    else: 
         sig_paras = np.array([sig10_2, sig11_2, sig20_2, sig21_2, alpha, beta])         
    print(sig_paras) 
    sig_l_02_samp[t] = np.array([sig10_2, sig11_2, sig20_2, sig21_2,sig22_2, alpha,beta]) 
    sd = 0.35*np.ones(cov_len) 
    if t ==0: 
        if method =='ab_joint':
             sig10_2, sig11_2, sig20_2, sig21_2,sig22_2, alpha,beta, Sigma, log_post_Sig =  update_Sigma_AM_onestp(Y,  mu,  lat, degree,  cof_K, sig10_2, sig11_2,sig20_2,sig21_2, alpha,beta, Sigma, sd, mu_ast) 
        elif method == 'ab_ind':        
             sig10_2, sig11_2, sig20_2, sig21_2,sig22_2, alpha,beta, Sigma, log_post_Sig =  update_Sigma_AM_onestp_beta(Y,  mu,  lat, degree,  cof_K, sig10_2, sig11_2,sig20_2,sig21_2, alpha,beta, Sigma, sd, mu_ast) 
        elif method == 'xi2':
            sig10_2, sig11_2, sig20_2, sig21_2,sig22_2, alpha,beta, Sigma, log_post_Sig =  update_Sigma_AM_onestp2(Y,  mu,  lat, degree,  cof_K, sig10_2, sig11_2,sig20_2,sig21_2, alpha,beta, Sigma, sd, mu_ast)         
    else: 
        if method =='ab_joint':
            sig10_2, sig11_2, sig20_2, sig21_2,sig22_2, alpha,beta, Sigma, log_post_Sig = update_Sigma_AM_onestp(Y,  mu,  lat, degree,  cof_K, sig_l_02_samp[t,0],sig_l_02_samp[t,1],sig_l_02_samp[t,2],sig_l_02_samp[t,3],sig_l_02_samp[t,5], sig_l_02_samp[t,6], Sigma, sd, mu_ast)
        elif method == 'ab_ind':
            sig10_2, sig11_2, sig20_2, sig21_2,sig22_2, alpha,beta, Sigma, log_post_Sig = update_Sigma_AM_onestp_beta(Y,  mu,  lat, degree,  cof_K, sig_l_02_samp[t,0],sig_l_02_samp[t,1],sig_l_02_samp[t,2],sig_l_02_samp[t,3],sig_l_02_samp[t,5], sig_l_02_samp[t,6], Sigma, sd, mu_ast)
        elif method == 'xi2':
            sig10_2, sig11_2, sig20_2, sig21_2,sig22_2, alpha,beta, Sigma, log_post_Sig = update_Sigma_AM_onestp2(Y,  mu,  lat, degree,  cof_K, sig_l_02_samp[t,0],sig_l_02_samp[t,1],sig_l_02_samp[t,2],sig_l_02_samp[t,3],sig_l_02_samp[t,5], sig_l_02_samp[t,6], Sigma, sd, mu_ast)
             
    Sig_y[t] = log_post_Sig    
 
#--------------------------------------------------------------------------------------------        
    sig30_2 =  psv.s_lm2(3,0,alpha,beta,sig10_2, sig11_2, sig20_2, sig21_2,sig22_2)
    if G4 == 0.0:
        sig_l_02 = np.array([sig10_2, sig20_2, sig30_2]) 
       
    else: 
        Sig40_2 = psv.s_lm2(4,0,alpha,beta,sig10_2, sig11_2, sig20_2, sig21_2,sig22_2)
        sig_l_02 = np.array([sig10_2, sig20_2, sig30_2, sig40_2]) 
      
    #print(Y[0])
    #update non-zonal effects G=[g_1^0, g_2^0, g_3^0] given Sig and R        
    sig_l_2_tild = []
    m_tild = []
    log_post_G = 0
    for j in range(len(G)):
        noJind = np.where(seqJ != j)[0]              
        mu_tild_cof = 0
        invsig_tild = 0
        for i in range(n):             
            GnoJ = G[noJind]           
            sumXGnotj =np.sum( X[i][:, noJind]*GnoJ.T, axis=1) #dim 2x1 #               
            XSig = np.linalg.solve(Sigma[i],X[i][:,j])   #Sigma^-1 X^T  dim = 2x1           
            XsigX = np.matmul(XSig.T, X[i][:,j]) #scalar
            XsigXGnoj = np.matmul(XSig.T, sumXGnotj) #scalar             
            XsigY = Y[i].dot(XSig)
            mu_tild_cof = mu_tild_cof - XsigXGnoj + XsigY
            invsig_tild = invsig_tild + XsigX    
        sig2 = 1/(invsig_tild + 1/100)  #flat prior
        #pdb.set_trace() 
        
        if sig2 < 0:
             sig2 = 0.0
        sig_l_2_tild.append(sig2)
        mu_j = sig2* mu_tild_cof
        m_tild.append(mu_j)   
        sig = np.sqrt(sig2)
# 
        tmp_g = np.random.normal(mu_j, sig)
        if np.isnan(tmp_g):
            tmp_g = 0.0
        G[j] = tmp_g                
        log_g = np.log(normal.pdf(tmp_g, loc=mu_j, scale=sig) )
        log_post_G = log_post_G + log_g
        
    G_samp[t] = G 
    print(G)    
    logpost_g[t] = log_post_G  
    new_post = log_post_Sig  - log_post_G      
    log_fullpost[t] =  new_post    
    if (t > nMCMC): 
        break    
    t += 1 
    print(t)
    

MCpost_G =  np.mean(G_samp[burnin:-1],axis=0)
MCpost_sig_l_02 = np.mean(sig_l_02_samp[burnin:-1], axis=0)    

end = time.time()
runtime = end - start
print(runtime)
    
    
    

Filename=directory + "/" + rstname  +"."+fileformat
with open(Filename,  'wb') as f: 
    pickle.dump([Y, X, R, cof_K, MCpost_G,MCpost_sig_l_02, G_samp,log_fullpost, sig_l_02_samp, Sig_y, logpost_g, runtime], f)
