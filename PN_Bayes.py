#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 11:32:08 2020

@author: jialiu
""" 
 
import numpy as np 
from scipy.stats import multivariate_normal
from scipy.stats import invwishart 
from scipy.stats import wishart  
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
from PNcov import *
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
#we use some function in PSV_dir_predictions package, there are two ways to call the function
#either install it or save it in some where--- /user_path/PSV/BCE19-dirPSV ---and call it in the script.
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

     
def update_Sigma(Y,mu, lat, degree, a_sig,  b_sig10, b_sig11, b_sig20,  b_sig21, a_beta, b_beta, lo_alpha,hi_alpha,  threshold):    
    n = len(Y)
    Sigma = np.zeros((n,3,3))
    def log_inverse_gamma(x, a,b):
        log_ig = (-a-1)*np.log(x) - b/x
        return log_ig
    
    log_post_old = 0
    go_on = 1
    count = 0
    n = 0.5*n
    sumlik = 0
    for i in range(len(Y)):
        sumlik = sumlik + np.matmul(Y[i]-mu[i], (Y[i]-mu[i]).T)
        #a= np.matmul(Y[i]-mu[i], (Y[i]-mu[i]).T)
        print(sumlik,i)
    lik = 0.5*sumlik
    while go_on:
        shape_sig = a_sig + n 
        
        scale_sig10 = b_sig_10 +lik
        sig_10_2 =  invgamma.rvs(shape_sig, scale=scale_sig10)      
        
        shape_sig =  b_sig_11 + n 
        scale_sig11 = b_sig_11 +lik
        sig_11_2 =  invgamma.rvs(shape_sig, scale=scale_sig11) 
        
        shape_sig =  b_sig_20 + n 
        scale_sig20 = b_sig_20 +lik
        sig_20_2 =  invgamma.rvs(shape_sig, scale=scale_sig20)
        
        shape_sig =  b_sig_21 + n 
        scale_sig21 = b_sig_21 +lik
        sig_21_2 =  invgamma.rvs(shape_sig, scale=scale_sig21)
        
        
        
#        sig_10_2 =  truncnorm.rvs(a_sig_10,  b_sig10)  
#        sig_11_2 =  truncnorm.rvs(a_sig_11,  b_sig11) 
#        sig_20_2 =  truncnorm.rvs(a_sig_20,  b_sig20)
#        sig_21_2 =  truncnorm.rvs(a_sig_21,  b_sig21)
        
        sig_22_2 = sig_20_2
        
        shape_beta = a_beta + n 
        scale_beta = b_beta +lik
        #beta =  invgamma.rvs(shape_beta, scale=scale_beta)
        beta = truncnorm.rvs(a_beta,b_beta)
        alpha = truncnorm.rvs(lo_alpha,hi_alpha)
        
        log_post0 = 0
        log_iv_sig20_2 = log_inverse_gamma(sig_20_2 , shape_sig, scale_sig20)
        log_post0 = log_post0 +  log_inverse_gamma(sig_10_2 , shape_sig,scale_sig10) +  log_inverse_gamma(sig_11_2 , shape_sig,scale_sig11) 
        log_post = log_post0 +   log_inverse_gamma(sig_21_2 , shape_sig,scale_sig21) +  2*log_iv_sig20_2 + log_inverse_gamma(beta , shape_beta, scale_beta)
        #print(log_post)
         
        count += 1        
        if (np.abs(log_post -  log_post_old)  < threshold ) or (count > 3000):
            go_on = 0
            lat0 = 0
           
            for i in range(len(Y)):
                if lat[i] != lat0:  
                    Sigma_i = psv.Cov(alpha,beta,lat[i], degree,sig_10_2,sig_11_2,sig_20_2,sig_21_2,sig_22_2)
    #                Sigma_i[0,1]=Sigma_i[0,2]
    #                Sigma_i[1,0] =Sigma_i[0,1]
    #                Sigma_i[0,2] = 0
    #                Sigma_i[2,0] = 0
                #sumloglik = sumloglik + np.log(multivariate_normal.pdf(Y[i], mean= mu[i] , cov=Sigma_i))
                    lat0 =lat[i]
                    Sigma[i] = Sigma_i            
                else:
                    Sigma[i] = Sigma[i-1]
        else:
              log_post_old = log_post
    return sig_10_2, sig_11_2, sig_20_2, sig_21_2,sig_22_2, alpha,beta,Sigma, log_post
        
    
    
    
    
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
    
"""
	Returns a map of the density function su from Khokhlov et al 2013
	GGPmodel is a dictionary with the parameters of a zonal GGP
	degree - is the degree in which the covariance is calculated
	dx and dy are the space in X and Y axes of the equal area projection
	hem is the hemisphere side (1 means positive, -1 means negative) 
"""    
    
def su_PN(m, Cov,dx,dy,hem):

    Lamb = np.linalg.inv(Cov)
    
    m_norm = np.sqrt(np.dot(np.matmul(Lamb,m),m))
    
    n = np.int(1/dx)
    X = np.arange(-n*dx,n*dx+dx,dx)
    
    n = np.int(1/dy)
    Y = np.arange(-n*dy,n*dy+dy,dy)
    
    XX,YY = np.meshgrid(X,Y)
    s = np.zeros(np.shape(XX))
    
    for i in range(np.shape(XX)[0]):
        for j in range(np.shape(XX)[1]):
            if (XX[i,j]**2 + YY[i,j]**2)<=1 :
                s[i,j] = su(XX[i,j],YY[i,j],hem,Lamb,m_norm,m)
    
    return X,Y,XX,YY,s   






def PSV_plotmap(sp, sn,GGP,lat,dx,dy,hem):
    name = GGP['name']
    if hem == 1:
        hem_name = 'Positive hems.'
    else:
        hem_name = 'Negative hems.'
    n = np.int(1/dx)
    X = np.arange(-n*dx,n*dx+dx,dx)
    
    n = np.int(1/dy)
    Y = np.arange(-n*dy,n*dy+dy,dy)


    XX,YY = np.meshgrid(X,Y)
    minp = np.min(sp)
    minn = np.min(sn)
    if minp<minn:
        minimo = minp
    else:
        minimo = minn
    maxp = np.max(sp)
    maxn = np.max(sn)
    if maxp>maxn:
        maximo = maxp
    else:
        maximo = maxn
    fig = plt.figure(figsize=[13,6], dpi=80)
            
    plt.subplot(121)
    xb=np.arange(-1,1.01,0.01)
    ybn=-np.sqrt(abs(1-xb**2))
    ybp=np.sqrt(abs(1-xb**2))
    plt.plot(xb,ybn, '--',c='0.5')
    plt.plot(xb,ybp,'--',c='0.5')
    plt.plot(0,0,'+',c='k')
    plt.axis('off', aspect='equal')
    plt.contour(X,Y,sp,levels=np.linspace(minimo,maximo,8), zorder = 5020)
    plt.text(0.0,0.92,'Lat=%i$^\circ$'%lat,horizontalalignment='center')
    plt.text(0.0,0.85,name + ' '+ hem_name,horizontalalignment='center')
    plt.text(0.0,0.78,'MCMC inclination', horizontalalignment='center')
                
         
    plt.subplot(122)
    plt.plot(xb,ybn, '--',c='0.5')
    plt.plot(xb,ybp,'--',c='0.5')
    plt.plot(0,0,'+',c='k')
    plt.axis('off', aspect='equal')
    plt.contour(X,Y,sn,levels=np.linspace(minimo,maximo,8),zorder = 5020)
    plt.text(0.0,0.92,'Lat=%i$^\circ$'%lat,horizontalalignment='center')
    plt.text(0.0,0.85,name + ' '+ hem_name,horizontalalignment='center')
    plt.text(0.0,0.78,'True inclination', horizontalalignment='center')         
     
    plt.show()

   
    
    
  
from scipy.stats import multivariate_normal

def simuPN_GGP(GGPmodel,lat):
    
    g10 = GGPmodel['g10']
    g20 = GGPmodel['g20']
    g30 = GGPmodel['g30']
    sig10 = GGPmodel['sig10']
    sig11 = GGPmodel['sig11']
    sig20 = GGPmodel['sig20']
    sig21 = GGPmodel['sig21']
    sig22 = GGPmodel['sig22']
    
    alpha = GGPmodel['alpha']
    beta = GGPmodel['beta']
    n = len(lat)
    u = np.zeros((n,3), dtype=float)
    r = np.zeros((n,1), dtype=float)
    for i in range(n):
        Sigma_i = psv.Cov(alpha,beta,lat[i], degree,sig10**2,sig11**2,sig20**2,sig21**2,sig22**2)  
        mu_i = psv.m_TAF(GGPmodel, lat[i])
        x = multivariate_normal.rvs(mu_i.tolist(), Sigma_i.tolist())        
        #u[i] = multivariate_normal([mu_i, Sigma_i]) 
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
    
        
#XI = simuXI(x0,0.5,n)       

def log_mnorm(m,Sig,xi):
    cov_len = len(xi)
    u = xi - m
    try:
        a = np.linalg.solve(Sig,u)
    except:
        Sig = Sig + np.diag(np.ones(cov_len))*1e-5
        a = np.linalg.solve(Sig,u)
    return -0.5*(np.log(np.linalg.det(Sig)) +  np.dot(u.T,a))    


def simuXI_y(x0,var,n,u,cof_K):
    #number of samples ----n
    q = len(x0) 
    XI= np.zeros((q,n))
    XIh= np.zeros((q,n))
    for j in range(q):
        low_x0 = x0[j] - var*x0[j]
        hi_x0 = x0[j] + var*x0[j]
        xi_j = np.random.uniform(low_x0,hi_x0,size = n)        
        XI[j] =  xi_j
    #h = np.zeros((n,1)) 
    #pos = np.zeros((n,1)) 
    meanXI = np.mean(XI,axis=1)
    XI2 = np.zeros((n,q))
    Cov = 0
    for i in range(n):
        sig_paras = np.array([XI[0,i],XI[1,i],XI[2,i],XI[3,i],XI[4,i]])
        #h[i] = log_post_fun(u, cof_K, sig_paras) + log_mnorm(meanXI,Sig,sig_paras)
        #Cov_i =  np.matmul((sig_paras-meanXI).reshape(q,1), (sig_paras-meanXI).reshape(1,q))
        #pos[i] = log_mnorm(meanXI,Sig,sig_paras)
        XI2[i] = sig_paras
        Cov = Cov +   np.matmul((sig_paras-meanXI).reshape(q,1), (sig_paras-meanXI).reshape(1,q))
        #var = var + (sig_paras-meanXI)**2
    Cov = 1/n *Cov
    #var = 1/n *var
    Var = np.diag(np.diag(Cov))
    #pr = h/np.sum(h)
    return XI2, meanXI, Cov, Var

import math 

def simuXI_y2(x0,var,n,u,cof_K):
    def log_mag_xi(cof_K,sig_paras,u):
        K = 0        
        sumlik = 0
        for j in range(len(sig_paras)): #covariates j                          
            K = K + sig_paras[j]* cof_K[j]    

        try:
             a = np.linalg.solve(K,u)
        except:
            K = K + np.diag(np.ones(len(u)))*1e-5    
            a = np.linalg.solve(K,u)      
        sumlik = sumlik + np.dot(u.T, a)  + np.log(np.linalg.det(K) )  
        return -0.5*sumlik     
    q = len(x0) 
    XI= np.zeros((q,n))
    XIh= np.zeros((q,n))
    
    for j in range(q):
        low_x0 = x0[j] - var*x0[j]
        hi_x0 = x0[j] + var*x0[j]
#        go_on = 1
#        while go_on:
        xi_j = np.random.uniform(low_x0,hi_x0,size = n)  
#            if xi_j > 0:
#                go_on =0
        XI[j] =  xi_j
    h = np.zeros((n,1))     
    #pos = np.zeros((n,1)) 
    meanXI = np.mean(XI,axis=1)
    XI2 = np.zeros((n,q))
    Cov = 0
    for i in range(n):
        sig_paras = np.array([XI[0,i],XI[1,i],XI[2,i],XI[3,i],XI[4,i]])        
        Cov_i =  np.matmul((sig_paras-meanXI).reshape(q,1), (sig_paras-meanXI).reshape(1,q))
        #h[i] =  log_mag_xi(cof_K[:,i],sig_paras,u[i])# + log_mnorm(meanXI,Cov_i,sig_paras)  #np.ones(q),
        print(np.exp(log_post_fun(u, cof_K, sig_paras)/np.prod(sig_paras)))
        #pos[i] = log_mnorm(meanXI,Sig,sig_paras)
        XI2[i] = sig_paras
        Cov = Cov +  Cov_i
        #var = var + (sig_paras-meanXI)**2
    Cov = 1/n *Cov
    #var = 1/n *var
    
    vInd = np.logical_not(np.isnan(h))
    h = h[vInd]
    vInd2 = vInd.reshape(1,n)[0]
    XI2 = XI2[vInd2] 
    #meanXI = np.mean(XI2,axis=0)
    #Cov = 1/len(XI2) *Cov[vInd2]
    #len(h)
    pr = h/np.sum(h) 
    go_on = 1
    while go_on:
        ur = np.random.uniform(0,1)
        distInd = np.argmin(np.sqrt((ur-pr)**2))    
    #np.delete(a, 1,axis=0)
        pt = XI2[distInd]
        if np.all(pt[0:len(pt)-1] > 0.0) and (pt[-1] >= 10):
            go_on = 0  
            empr = pr[distInd]
    return XI2, h, pt, meanXI, Cov, empr




 
 

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

 
 
def ndetK(cof_K,sig_paras):
    logK = 0 
    A  = np.log(cof_K)   #log(det(K)) = tr(log(K))
    sumA = np.sum(A,axis=1)
   
    n = cof_K.shape[1]
    for i in range(sumA.shape[0]):
        trA = np.matrix.trace(sumA[i])
        #print(trA)
        if (not np.isnan(trA)) and (not np.isinf(trA)):
            logK = logK -0.5  * trA - 0.5*3*n * np.log(sig_paras[j])
            print(logK)
    return logK
     
        
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

def simuPN_GGP(GGPmodel,lat,k):   
   
    g10 = GGPmodel['g10']
    g20 = GGPmodel['g20']
    g30 = GGPmodel['g30']
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


  

def AMcovMirror(meanXIt, x0, C0, C_t, t, meanCovt_1 ):
    
    def get_near_psd(A):
        C = (A + A.T)/2
        eigval, eigvec = np.linalg.eig(C)
        eigval[eigval < 0] = 0
        return eigvec.dot(np.diag(eigval)).dot(eigvec.T)
    
    if cov_len  ==5:          
         mu_ast = 2*np.array([40.96,  2.89,  0.36,  4.84, 56.25])
    else:        
        mu_ast = 2*np.array([40.96,  2.89,  0.36,  4.84, 56.25,14.4])

    epsilon = 1e-5
    q = len(x0)
    sd =  1
    if t ==0:
        covXI = C0 + sd * np.identity(q)*epsilon  
        meanCovt = 0
    elif t==1:              
        #meanXIt = np.mean(Tsigparas,axis=0)
        meanCovt  = meanXIt.reshape(q,1)* meanXIt.reshape(1,q)
        #K = Tsigparas[0].reshape(q,1)* Tsigparas[0].reshape(1,q) + Tsigparas[1].reshape(q,1)* Tsigparas[1].reshape(1,q)
        K = meanCovt_1 + x0.reshape(q,1)* x0.reshape(1,q)
        covXI = K - 2* meanCovt 
        covXI =  sd*covXI + sd * np.identity(q)*epsilon        
    else:
 
        meanCovt = meanXIt.reshape(q,1)* meanXIt.reshape(1,q)            
        covXI = (t-1)/t* C_t + sd/t*(t* meanCovt_1 - (t+1)* meanCovt + x0.reshape(q,1)* x0.reshape(1,q) + np.identity(q)*epsilon)       
   
    if np.any(np.linalg.eigvals(covXI) < 0):
        covXI = get_near_psd(covXI) +  np.identity(q)*1e-5
        #pdb.set_trace()
    
    covXI = 0.25*covXI
    go_on = 1
    while go_on:
        sampXI = multivariate_normal.rvs(mean = 2*mu_ast  - x0, cov = covXI, size = 1)                
        if np.all(sampXI > 1e-4):
        
            go_on = 0
  
    return sampXI, covXI, meanCovt 




def Omega_mat(Sigma):
    det_Sigma_ast =1/( Sigma[0,0]*Sigma[2,2] - Sigma[0,2]**2)
     
    Omega_mat = np.zeros([3,3])
    Omega_mat[0,0] = det_Sigma_ast*Sigma[2,2]
    Omega_mat[0,2] = -det_Sigma_ast*Sigma[0,2]
    Omega_mat[1,1] = 1/Sigma[1,1]
    Omega_mat[2,0] = Omega_mat[0,2]
    Omega_mat[2,2] = det_Sigma_ast*Sigma[0,0]
    return Omega_mat

  

def VGP_trans(x,site_lat,site_lon):
    #x--has a cartesian coordinates
    # output is vgp transformed data
    
    #thetaY,phiY=polarc(x) #incl and azimuth 
    #    return np.array([Decs, Incs, Rs]).transpose()      
    DIR = pmag.cart2dir(x) 
    vgp = pmag.dia_vgp(DIR[:,0],DIR[:,1],1,site_lat,site_lon)  
    vgp = np.array(vgp)
    vgp_lon = vgp[0,:][:,np.newaxis]
    vgp_lat = vgp[1,:][:,np.newaxis]
    DI = np.column_stack((vgp_lon,vgp_lat))
    vgp_x= pmag.dir2cart(DI)
    vgp_dp = vgp[2,:][:,np.newaxis]
    vgp_dm = vgp[3,:][:,np.newaxis]
    PM = np.column_stack((vgp_dp,vgp_dm))
    return vgp_x, PM
    
    
