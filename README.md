# Bayes_GGP----implements a Bayesian approach to estimate the time-average geomagnetic field mean and paleosecular variation and their uncertainties based on a projected normal distribution. 
The proposed method can directly work with paleomagnetic directional data without reference to experimental intensities.
Bayes_GGP is written in Phython and contains two files: main.py and PN_Bayes.py
(1) main.py includes the Bayesian regression and estimation framework using the MCMC method, and three sub functions that contain three different choices to update the variation parameters, \bm xi.  
(2) Py_Bayes.py includes the data augmentation approach to update the intensities, and other functions to support the main scheme.   
