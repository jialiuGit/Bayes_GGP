#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 14:50:55 2020

@author: jialiu

This code is modified from some functions in PSV_dir_predictions. The aim is to separate 
alpha and beta in the variance function of Gauss coefficients, \sigma_l^m
"""
import numpy as np 
import sys
sys.path.insert(1, '/user_path/PSV/BCE19-dirPSV')
import PSV_dir_predictions as psv 
import math


def PNs_lm2(l,m,alpha,tau,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2):
	
    """
    Variance each gauss coefficient 
    l,m are the degree
    alpha is the alpha factor since CP88
    tau = alpha*beta
    sig10_2 is the squared standard deviation for g10, which for CP88 is different
    sig10 = 3 then sig10_2=3*3=9 for CP88 model
    if sig10_2 is zero, then it will be calculated by the same equation as non-dipolar coefficients
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
    The variance in Phi direction (through the East-West direction) - propagation of error following that each component of the magenetic field
    is a linear combination of the gauss coeficients glm and hlm
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
    Calculates the covaricance between Br and Btheta"""
	sum_l = 0.
	for i in range(1,l+1):
		A = PNs_lm2(i,0,alpha,tau,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2)*psv.P_lm(i,0,theta)*psv.dP_lm_dt(i,0,theta)
		sum_m = 0.
		for j in range(1,i+1):
			B = (math.factorial(i-j)/math.factorial(i+j))*PNs_lm2(i,j,alpha,tau,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2)*psv.P_lm(i,j,theta)*psv.dP_lm_dt(i,j,theta)
			sum_m += B
		sum_l = sum_l -(i+1)*(A+2*sum_m)
	return sum_l 
#import pdb
def PNCov(alpha,tau,lat, degree,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2):
    """
    Covariance Matrix for a given GGP model in a latitude = lat in degrees
    degree is the maximum degree of gaussian coeficients"""
    theta = np.deg2rad(90-lat)
    Cov = np.zeros([3,3])
    #pdb.set_trace()
    Cov[0,0] = PNsig_bt2(degree,alpha,tau,theta ,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2)     
    Cov[0,2] = PNcov_br_bt(degree,alpha,tau,theta ,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2) 
    Cov[1,1] = PNsig_bph2(degree,alpha,tau,theta ,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2)
    Cov[2,0] = PNcov_br_bt(degree,alpha,tau,theta ,sig10_2,sig11_2,sig20_2,sig21_2,sig22_2) 
    Cov[2,2] = PNsig_br2(degree,alpha,tau,theta, sig10_2,sig11_2,sig20_2,sig21_2,sig22_2)
    
    return Cov


