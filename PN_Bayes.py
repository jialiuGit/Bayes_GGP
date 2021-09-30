#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 11:32:08 2020
@author: jialiu
""" 
 
import numpy as np 
from scipy.stats import multivariate_normal
import math
from scipy.stats import invgamma
from scipy.stats import gamma
from scipy.stats import truncnorm
from scipy.stats import norm as normal
from scipy.optimize import minimize as minimize
from numpy.random import dirichlet as dirichlet
from collections import Counter
from numpy.linalg import norm as norm
from scipy.optimize import lsq_linear
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import matplotlib.legend as legend
import pmagpy.ipmag as ipmag
import pmagpy.pmag as pmag
import numpy as np 
from random import randint
from scipy.stats import multinomial
import pdb

import sys
sys.path.insert(1, '/user_path/PSV/BCE19-dirPSV')
"""
we call some functions in pmagpy (Tauxe, L. et al., 2016) and PSV_dir_predictions (Brandt et al. 2020)  software packages, there are two ways do so
either install the software package or save it in some where--- /user_path/PSV/BCE19-dirPSV ---and call it from its path.

We chose the 2nd way to call the functions in PSV_dir_predictions, since we made some modifications of one function s_lm2, see below:
    
    --------------------------------------------------------------
def s_lm2(l,m,alpha,beta,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2):
    
    c_a = 0.547
    if ((l-m)/2. - int((l-m)/2)) == 0:
       
        s_lm2 = ((c_a**(2*l))*(alpha**2))/((l+1)*(2*l+1))
        
    else:        
        s_lm2 = (c_a**(2*l))*((alpha*beta)**2)/((l+1)*(2*l+1))
       
    if (l==1 and m==0):
        if (sig10_2>=0):    #----------allowed it to be zero in order to calculate those K_l^m terms by cof_K1, cof_K2 and cof_Sigma
            #print('sig10=%.2f' % np.sqrt(sig10_2))
            s_lm2 = sig10_2
    if (l==1 and m==1):
        if (sig11_2>=0):    #----------allowed it to be zero in order to calculate those K_l^m terms by cof_K1, cof_K2 and cof_Sigma
            #print('sig11=%.2f' % np.sqrt(sig11_2))
            s_lm2 = sig11_2
    if (l==2 and m==0):
        if (sig20_2>=0):    #----------allowed it to be zero in order to calculate those K_l^m terms by cof_K1, cof_K2 and cof_Sigma
            #print('sig20=%.2f' % np.sqrt(sig20_2))
            s_lm2 = sig20_2
    if (l==2 and m==1):
        if (sig21_2>=0):    #----------allowed it to be zero in order to calculate those K_l^m terms by cof_K1, cof_K2 and cof_Sigma
            #print('sig21=%.2f' % np.sqrt(sig21_2))
            s_lm2 = sig21_2
    if (l==2 and m==2):
        if (sig22_2>=0):    #----------allowed it to be zero in order to calculate those K_l^m terms by cof_K1, cof_K2 and cof_Sigma
            #print('sig22=%.2f' % np.sqrt(sig22_2))
            s_lm2 = sig22_2     
    return s_lm2    
    
"""

import PSV_dir_predictions as psv 
import Synthetic_directions_GGP as simGGP 

def slice_samp_r_weight(r0, u,mu,K,p):
    a = np.linalg.solve(K,u)
    b = np.linalg.solve(K,mu)
    a = np.dot(u.T, a)
    b = np.dot(u.T, b)
    
    def f(r):  
        return np.exp(-0.5*a*(r-b/a)**2)    
    nv = np.random.uniform(0, f(r0))  
    nv1 = -2*np.log(nv)/a
    u1 = np.random.uniform(0, 1)  
    if nv1 < 0.0 or np.isinf(nv1):
       nv1 = 0.0              
    tem1 = b/a
    tem2 = np.sqrt(nv1)
    g_lo = tem1 + np.max([-tem1, -tem2])
    g_hi = tem1 + tem2
    r =  np.power( (g_hi**p -g_lo**p)*u1 +  g_lo**p, 1/p)   
    if np.isnan(r):
        r=1.0
    return r
 
  
def KLdist(A,B):
    KLdist = np.trace(np.linalg.solve(A,B)) - np.log(np.linalg.det(B)/np.linalg.det(A))
    return KLdist

   
"""
Construct design matrix X from the zonal field for a given latitude and returns the horizontal and vertical components Btetha, Br
where a is the Earth radius and r is the distance you want to calculate from the center of earth
a_r = a/r
for example: a_r = 1.0 for the surface of Earth

"""

def deg_mat_zonal(lat,G4,a_r=1.0): #correct 07.1.2021  
    Theta = 90-lat  
    
    costheta = np.cos(np.deg2rad(Theta))
    sintheta = np.sin(np.deg2rad(Theta))
    
    Br1 = -2*(a_r**3)*costheta
    Br2 = -(3/2)*(a_r**4)*(3*(costheta**2) - 1) 
    Br3 = -(a_r**5)*2*(5*(costheta**3) - 3*costheta)
    Br4 = -(a_r**6)*(5/8)*(35*(costheta**4) - 30*(costheta**2)+3)
    
    
    Bt1 = -(a_r**3)*sintheta
    Bt2 =  -(a_r**4)*3*sintheta*costheta
    Bt3 = -(a_r**5)/2*(15*sintheta*(costheta**2) - 3*sintheta)
    Bt4 = -(a_r**6)/2*(35*sintheta*(costheta**3) - 15*sintheta*costheta)
    
    if G4==0.0:
        X= np.array([[Bt1,Bt2,Bt3],[Br1,Br2,Br3]])
    else:        
        X= np.array([[Bt1,Bt2,Bt3,Bt4], [Br1,Br2,Br3,Br4]])
    return X
    
 

def sim_GGP_data(GGPmodel,lat,degree,k):    
    if k!=0:
        lat_n = np.repeat(lat,k)
    else:
        lat_n = lat
    inten = []   
    for i in range(len(lat_n)):        
         one_pt = simGGP.dii_fromGGPmodels(1,lat_n[i],GGPmodel, degree=degree)
         dir_onept = pmag.dir2cart([ one_pt[0][0],one_pt[0][1]])  
         if i == 0:
            GGPdata = dir_onept             
         else:
            GGPdata = np.vstack((GGPdata, dir_onept))
         inten.append(one_pt[0][2]) 
    return GGPdata, inten, lat_n
     
   

def simuPN_GGP(GGPmodel,lat, degree, k):
    
#    g10 = GGPmodel['g10']
#    g20 = GGPmodel['g20']
#    g30 = GGPmodel['g30']
    sig10 = GGPmodel['sig10']
    sig11 = GGPmodel['sig11']
    sig20 = GGPmodel['sig20']
    sig21 = GGPmodel['sig21']
    sig22 = GGPmodel['sig22']
    
    alpha = GGPmodel['alpha']
    beta = GGPmodel['beta']     
    
    if k!=0:
        lat_n = np.repeat(lat,3)
    else:
        lat_n = lat
    n = len(lat_n)
    u = np.zeros((n,3), dtype=float)
    r = np.zeros((n,1), dtype=float)
    for i in range(n):
        Sigma_i = psv.Cov(alpha,beta,lat_n[i], degree,sig10**2,sig11**2,sig20**2,sig21**2,sig22**2)  
        mu_i = psv.m_TAF(GGPmodel, lat_n[i])
        x = multivariate_normal.rvs(mu_i.tolist(), Sigma_i.tolist())      
        x_len = np.linalg.norm(x)
        u[i] = x/x_len 
        r[i] = x_len
    return u,r
    
    
         
#simulate xi
def simuXI(x0,var,n):
    q = len(x0) 
    XI= np.zeros((q,n+1))
    XIh= np.zeros((q,n+1))
    for j in range(q):
        low_x0 = x0[j] - var*x0[j]
        hi_x0 = x0[j] + var*x0[j]
        xi_j1 = np.random.uniform(low_x0,x0[j],size = n+1)
        xi_j2 = np.random.uniform(x0[j],hi_x0,size = n+1)
        XI[j] =  xi_j1 
        XIh[j] = xi_j2
    XIh2 =  XIh + var*XIh 
    XI2 =  XI - var*XI 
    return XI,XIh, XIh2,XI2

def log_mnorm(m,Sig,xi):
    cov_len = len(xi)
    u = xi - m
    try:
        a = np.linalg.solve(Sig,u)
    except:
        Sig = Sig + np.diag(np.ones(cov_len))*1e-5
        a = np.linalg.solve(Sig,u)
    return -0.5*(np.log(np.linalg.det(Sig)) +  np.dot(u.T,a))    

def nSigma(Y,lat,sig_paras,degree=8):
    n=len(Y)
    sig10_2, sig11_2,sig20_2,sig21_2, alpha, beta = sig_paras 
    Sigma = np.zeros((n,3,3))
    detSig = np.zeros((n,1))
    lat0 =0
    for i in range(n):
                if lat[i] != lat0:  
                    Sigma_i = psv.Cov(alpha,beta,lat[i], degree,sig10_2,sig11_2,sig20_2,sig21_2,sig20_2)  
                    lat0 =lat[i]
                    Sigma[i] = Sigma_i  
                    detSig[i] = np.linalg.det(Sigma_i)
                else:
                    Sigma[i] = Sigma[i-1]
                    detSig[i] =  detSig[i-1]  
    return Sigma, detSig

def log_post_fun(u, cof_K, sig_paras):

    n = len(u)
    cov_len = cof_K.shape[0]
 
    sumlik = 0
 
    for i in range(n):    #This step can be done in parallel! line 503--513         
        K = 0
        for j in range(cov_len): #covariates j                          
            K = K + sig_paras[j]* cof_K[j][i]
        Kpre = Omega_mat(K)      
        A = u[i].reshape(3,1)*u[i].reshape(1,3)
        unew = A         
        V = np.dot(unew, Kpre)          
        sumlik = sumlik + np.matrix.trace(V)   + np.log(np.linalg.det(K) )          
    lik = -0.5*sumlik  
    return lik


def log_post_funIG(u, cof_K, sig_paras, a1, b1):
#place an exponential prior
    n = len(u)
    cov_len = cof_K.shape[0]
     
    sumlik = 0
    for i in range(n):    #This step can be done in parallel! l      
        K = 0
        for j in range(cov_len): #covariates j                          
            K = K + sig_paras[j]* cof_K[j][i]
        Kpre = Omega_mat(K)
        A = u[i].reshape(3,1)*u[i].reshape(1,3)
        unew = A #+ n* K        inverse exponential prior      
        V = np.dot(unew, Kpre)         
        sumlik = sumlik + np.matrix.trace(V)   + np.log(np.linalg.det(K) )   
    lik = -0.5*sumlik 
     
    def log_inverse_gamma(x, a,b):
        log_ig = 0
        for j in range(len(x)):
            log_ig = log_ig - ((a[j] + 1)*np.log(x[j]) + b[j]/x[j])
        return log_ig 
    a1= 0.01
    b1 =0.01
    if cov_len == 5:
        lik = lik + log_inverse_gamma(sig_paras, np.ones(5)*a1,np.ones(5)*b1) 
    else: 
        lik = lik + log_inverse_gamma(sig_paras, np.ones(6)*a1,np.ones(6)*b1)    
    return lik


def log_lik_diff(u, cof_K, sig_paras, x0):
    n = len(u)
    cov_len = cof_K.shape[0] 
    diffloglik = 0
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
    lik = 0.5*diffloglik 
    return lik

        
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0) 
 
def multivariate_student_logp(X, mu, Sig, df):    
    #multivariate student T distribution
    cov_len = len(X)
    u = X - mu
    try:
        a = np.linalg.solve(Sig,u)
    except:
        Sig = Sig + np.diag(np.ones(cov_len))*1e-5
        a = np.linalg.solve(Sig,u) 
    return -0.5*np.log(np.linalg.det(Sig)) - 0.5*(df+cov_len)* np.log(1+ np.dot(u.T,a))    


def cof_Sigma(max_degree,lat):    # compute K2    
    cof_K = np.zeros((6,3,3))     
    degree = np.array([1,1,2,2,max_degree,max_degree])
    cof = np.array(np.zeros(6), dtype=float) 
    for i in range(6):   
         cof = np.array(np.zeros(6), dtype=float) 
         if (i ==4): #for alpha
             cof = np.array(np.zeros(6), dtype=float) 
             cof[i] = 1.0   #alpha^2, beta^2
             cof[i+1] = 3.8#             
         else:
            cof = np.array(np.zeros(6), dtype=float) 
            cof[i] = 1.0
            sig10_2,sig11_2,sig20_2,sig21_2,alpha,beta = cof 
         sig22_2 = 0  
         if i ==5:
              Cov = PNCov(alpha,beta,lat,degree[i],sig10_2,sig11_2,sig20_2,sig21_2,sig22_2)  
         else:
              Cov = psv.Cov(alpha,beta,lat,degree[i],sig10_2,sig11_2,sig20_2,sig21_2,sig22_2)   
         cof_K[i] = Cov #+ np.diag((1,1,1))*1e-5  # jitterSigma2    
    return cof_K
 

   
 
def cof_K1(degree, lat, fix_beta): 
#Return 5 outputs each contains n*3*3 matrix
#This function only works when beta is fixed!     
    cof = np.array(np.zeros(6), dtype=float)    
    n= len(lat)   
    cof_K0 = np.zeros((n,3,3))
    cof_K1 = np.zeros((n,3,3))
    cof_K2 = np.zeros((n,3,3))  
    cof_K3 = np.zeros((n,3,3))
    cof_K4 = np.zeros((n,3,3))      
    cof_K = np.zeros((5,n,3,3))
    for i in range(n):           
         cof_K0[i] = psv.Cov(cof[4],cof[5],lat[i],1,1.0,cof[1],cof[2],cof[3],cof[2])          
         cof_K1[i] = psv.Cov(cof[4],cof[5],lat[i],1,cof[0],1.0,cof[2],cof[3],cof[2])        
         cof_K2[i] = psv.Cov(cof[4],cof[5],lat[i],2,cof[0],cof[1],1.0,cof[3],1.0) 
         cof_K3[i] = psv.Cov(cof[4],cof[5],lat[i],2,cof[0],cof[1],cof[2],1.0,cof[2])   # --sig21_2         
         cof_K4[i] = psv.Cov(1.0,fix_beta,lat[i],degree,cof[0],cof[1],cof[2],cof[3],cof[2])         
    cof_K[0]  = cof_K0
    cof_K[1]  = cof_K1
    cof_K[2]  = cof_K2
    cof_K[3]  = cof_K3
    cof_K[4]  = cof_K4 
    return cof_K


def cof_K2(degree, lat): 
#consider beta is r.v.
#Return 6 outputs each contains n*3*3 matrix
#This function only works when beta is r.v.   
    cof = np.array(np.zeros(6), dtype=float)    
    n= len(lat)    
    cof_K0 = np.zeros((n,3,3))
    cof_K1 = np.zeros((n,3,3))
    cof_K2 = np.zeros((n,3,3))  
    cof_K3 = np.zeros((n,3,3))
    cof_K4 = np.zeros((n,3,3)) 
    cof_K5 = np.zeros((n,3,3))
         
    cof_K = np.zeros((6,n,3,3))
    for i in range(n):           
         cof_K0[i] = psv.Cov(cof[4],cof[5],lat[i],1,1.0,cof[1],cof[2],cof[3],cof[2])          
         cof_K1[i] = psv.Cov(cof[4],cof[5],lat[i],1,cof[0],1.0,cof[2],cof[3],cof[2])        
         cof_K2[i] = psv.Cov(cof[4],cof[5],lat[i],2,cof[0],cof[1],1.0,cof[3],1.0)  
         cof_K3[i] = psv.Cov(cof[4],cof[5],lat[i],2,cof[0],cof[1],cof[2],1.0,cof[2])   # --sig21_2          
         cof_K4[i] = psv.Cov(1.0,cof[5],lat[i],degree,cof[0],cof[1],cof[2],cof[3],cof[2])         
         cof_K5[i] = psv.Cov(1.0,1.0,lat[i],degree,cof[0],cof[1],cof[2],cof[3],cof[2]) -cof_K4[i]          
    cof_K[0]  = cof_K0
    cof_K[1]  = cof_K1
    cof_K[2]  = cof_K2
    cof_K[3]  = cof_K3
    cof_K[4]  = cof_K4    
    cof_K[5]  = cof_K5    
    return cof_K  


def Omega_mat(Sigma):
    det_Sigma_ast =1/( Sigma[0,0]*Sigma[2,2] - Sigma[0,2]**2)
     
    Omega_mat = np.zeros([3,3])
    Omega_mat[0,0] = det_Sigma_ast*Sigma[2,2]
    Omega_mat[0,2] = -det_Sigma_ast*Sigma[0,2]
    Omega_mat[1,1] = 1/Sigma[1,1]
    Omega_mat[2,0] = Omega_mat[0,2]
    Omega_mat[2,2] = det_Sigma_ast*Sigma[0,0]
    return Omega_mat

 


"""
The following functions are modified from some functions in PSV_dir_predictions. The aim is to separate 
xi_5 = alpha and xi_6 (tau) = alpha x beta in computing the field covariance.
"""

def PNs_lm2(l,m,alpha,tau,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2):
	
    """
    Variance each gauss coefficient 
    l,m are the degree
    alpha is the alpha factor since CP88
    tau = alpha*beta      
    Note that here sig_lm_2 (l=1,2) must strict positive in modification.
    """
    
    c_a = 0.547
    if ((l-m)/2. - int((l-m)/2)) == 0:        
        s_lm2 = ((c_a**(2*l))*(alpha**2))/((l+1)*(2*l+1))        
    else:        
        s_lm2 = (c_a**(2*l))*(tau**2)/((l+1)*(2*l+1))  # Change by Jia Liu 12.11.2020
       
    if (l==1 and m==0):
        if (sig10_2>0):
            #print('sig10=%.2f' % np.sqrt(sig10_2))
            s_lm2 = sig10_2
    if (l==1 and m==1):
        if (sig11_2>0):
            #print('sig11=%.2f' % np.sqrt(sig11_2))
            s_lm2 = sig11_2
    if (l==2 and m==0):
        if (sig20_2>0):
            #print('sig20=%.2f' % np.sqrt(sig20_2))
            s_lm2 = sig20_2
    if (l==2 and m==1):
        if (sig21_2>0):
            #print('sig21=%.2f' % np.sqrt(sig21_2))
            s_lm2 = sig21_2
    if (l==2 and m==2):
        if (sig22_2>0):
            #print('sig22=%.2f' % np.sqrt(sig22_2))
            s_lm2 = sig22_2        
    return s_lm2

def PNsig_br2(l,alpha,tau,theta,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2):
    """
    l is the maximum degree for calculating, theta in radians - co-latitude
    The variance in r direction - propagation of error following that each component of the magenetic field
    is a linear combination of the gauss coeficients glm and hlm
    tau = alpha*beta 
    """
    sum_l = 0
        
    for i in range(1,l+1):
        #print(i)
        A = ((i+1)**2)*PNs_lm2(i,0,alpha,tau, sig10_2,sig11_2,sig20_2,sig21_2,sig22_2)*(psv.P_lm(i,0,theta)**2)
#        import pdb
#        pdb.set_trace()
 
        #print(A)
        sum_m=0.
        for j in range(1,i+1): 
            #print (j)
            B = ((math.factorial(i-j))/(math.factorial(i+j)))*PNs_lm2(i,j,alpha,tau,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2)*psv.P_lm(i,j,theta)**2
            sum_m = sum_m + B
        #print(((i+1)**2)*2*sum_m)
        sum_l = sum_l + A + ((i+1)**2)*2*sum_m
    return sum_l

def PNsig_bt2(l,alpha,tau,theta,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2):
    """
    l is the maximum degree for calculating, theta in radians - co-latitude   
    The variance in Theta theta in radians - co-latitude)direction (through North-South direction) - propagation of error following that each component of the magenetic field
    is a linear combination of the gauss coeficients glm and hlm
    tau = alpha*beta 
    """
    sum_l = 0
   
    for i in range(1,l+1):
        #print(i)
        A = PNs_lm2(i,0,alpha,tau,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2)*(psv.dP_lm_dt(i,0,theta)**2)
        
        #print(A)
        sum_m=0.
        for j in range(1,i+1): 
            #print (j)
            B = ((math.factorial(i-j))/(math.factorial(i+j)))*PNs_lm2(i,j,alpha,tau,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2)*psv.dP_lm_dt(i,j,theta)**2
            sum_m = sum_m + B
        #print(((i+1)**2)*2*sum_m)
        sum_l = sum_l + A + 2*sum_m
    return sum_l
	
def PNsig_bph2(l,alpha,tau,theta,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2):
    """
    l is the maximum degree for calculating, theta in radians - co-latitude     
    tau = alpha*beta 
    """

    sum_l = 0
    if theta == 0:
        print('lat=90 will be aproximated to 89.999999')
        theta = np.deg2rad(90-89.999999)
    for i in range(1,l+1):
        sum_m=0.
        for j in range(1,i+1): 
            B = (j**2)*((math.factorial(i-j))/(math.factorial(i+j)))*PNs_lm2(i,j,alpha,tau,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2)*(psv.P_lm(i,j,theta)**2)
            sum_m = sum_m + B
        #print(((i+1)**2)*2*sum_m)
        sum_l = sum_l + 2*sum_m/(np.sin(theta)**2)
    return sum_l

def PNcov_br_bt(l,alpha, tau, theta,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2):
	"""
    l is the maximum degree for calculating, theta in radians - co-latitude
    Calculates the covaricance between Br and Btheta
    tau = alpha*beta 
    """
	sum_l = 0.
	for i in range(1,l+1):
		A = PNs_lm2(i,0,alpha,tau,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2)*psv.P_lm(i,0,theta)*psv.dP_lm_dt(i,0,theta)
		sum_m = 0.
		for j in range(1,i+1):
			B = (math.factorial(i-j)/math.factorial(i+j))*PNs_lm2(i,j,alpha,tau,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2)*psv.P_lm(i,j,theta)*psv.dP_lm_dt(i,j,theta)
			sum_m += B
		sum_l = sum_l -(i+1)*(A+2*sum_m)
	return sum_l 


def PNCov(alpha,tau,lat, degree,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2):
    """
    Covariance Matrix for a given GGP model in a latitude = lat in degrees
    degree is the maximum degree of gaussian coeficients
    tau = alpha*beta 
    """
    theta = np.deg2rad(90-lat)
    Cov = np.zeros([3,3])
    #pdb.set_trace()
    Cov[0,0] = PNsig_bt2(degree,alpha,tau,theta ,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2)     
    Cov[0,2] = PNcov_br_bt(degree,alpha,tau,theta ,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2) 
    Cov[1,1] = PNsig_bph2(degree,alpha,tau,theta ,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2)
    Cov[2,0] = PNcov_br_bt(degree,alpha,tau,theta ,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2) 
    Cov[2,2] = PNsig_br2(degree,alpha,tau,theta, sig10_2,sig11_2,sig20_2,sig21_2,sig22_2)    
    return Cov


