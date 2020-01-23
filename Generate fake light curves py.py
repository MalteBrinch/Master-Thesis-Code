#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import random
import matplotlib.pyplot as plt
import pymc3 as pm


# In[5]:


# set the seed
np.random.seed(42)# the random seed

n = 5000 #number of data points
start = 58000# start of observation
end = 58900# end of observation
tau = np.linspace(0.0, 100.0, n)# delay times for transfer functions
X = np.linspace(start, end, n)[:, None]# The time inputs to the driving, they must be arranged as a column vector

# Define the true covariance function and its parameters
ℓ_true = np.sqrt(1.0/2.0)#timescale of variation for the driving function
#REMEMBER time scale is 2*ℓ^2 so remember to rewrite as ℓ_true=2*ℓ^2
η_true = 0.15#long term standard deviation for the driving function
α_true = 1.0#exponent related to the PSD slope 
np.savetxt('hyperparameters.txt',np.c_[ℓ_true,η_true,α_true],delimiter=',')
cov_func = η_true**2 * pm.gp.cov.Exponential(1, 2.0**((α_true-1.0)/2.0)*ℓ_true**(α_true))#defined kinda funky due to pymc3

# A mean function that is zero everywhere
mean_func = pm.gp.mean.Zero()

# The latent function values are one sample from a multivariate normal
# Note that we have to call `eval()` because PyMC3 built on top of Theano
f_true = np.random.multivariate_normal(mean_func(X).eval(), cov_func(X).eval(), 1).flatten()
#f_true = np.random.multivariate_normal(mean_func(X).eval(),cov_func(X).eval() + 1e-8*np.eye(n), 1).flatten()
np.savetxt('drivingfunction.txt',np.c_[X,f_true],delimiter=',')
print('time/delta mag values are saved under "drivingfunction.txt". They are ordered as follows: [time,delta mag]')


# In[7]:


def transferDT(wav, pDT, tau):
    sigma_DT, m_DT, theta_DT, T = pDT

    h = 6.626e-34# Plancks constant in units of [m^2*kg/s]
    c = 299792458.0# speed of light in units of [m/s]
    k = 1.38e-23# Boltzmanns constant in units of [m^2*kg/s^2*K]
    wav_0 = 1122.4# Reference wavelength in nm, use 500?
    
    # peak Black Body from uniform torus temperature
    wav_peak = 2.898*10**6/T
    b_max = 4.967#h*c/(1e-9*wav_peak*k*T)
    BB_max = 1.0/( (wav_peak**5) * (np.exp(b_max) - 1.0) )
    
    # Universal lognormal for Dusty Torus 
    exp_DT = -((np.log((tau-theta_DT)/m_DT))**2/(2*sigma_DT**2)) 
    front_DT = 1.0/((tau-theta_DT)*sigma_DT*np.sqrt(2*np.pi))
    lognorm_DT = front_DT*np.exp(exp_DT)
    where_are_NaNs1 = np.isnan(lognorm_DT)
    lognorm_DT[where_are_NaNs1] = 0.0
    
    # Dusty Torus transfer equation for band
    b = h*c/(1e-9*wav*k*T)
    BB = (1.0/( wav**5 * (np.exp(b) - 1.0) ))/BB_max
    Psi_DT = BB*lognorm_DT
    
    return Psi_DT
    
def transferAD(wav, K_0, index, pAD, tau):
    sigma_AD, m_AD, theta_AD = pAD
    
    wav_0 = 1122.4# Reference wavelength in nm, use 500?
    
    # Accretion Disk transfer equation for the band
    powr = K_0*(wav/wav_0)**(index)    
    exp_AD = -((np.log((tau-theta_AD)/m_AD))**2/(2*sigma_AD**2))
    front_AD = 1.0/((tau-theta_AD)*sigma_AD*np.sqrt(2*np.pi))
    lognorm_AD = front_AD*np.exp(exp_AD)
    where_are_NaNs2 = np.isnan(lognorm_AD)
    lognorm_AD[where_are_NaNs2] = 0.0
    Psi_AD = powr*lognorm_AD

    return Psi_AD

# Function to convolve driving function with transfer function to obtain data
def createdata(f_true, tau, wav, K_0, index, pDT, pAD, noise_scale):
    Psi_tot = transferDT(wav, pDT, tau) + transferAD(wav, K_0, index, pAD, tau)
    data = np.convolve(f_true,Psi_tot,'same')
    uncer = np.random.rand(len(f_true))*noise_scale
    return [data,uncer]


# In[8]:


# create data for all bands
pDT = [2.4,39.0,37.0,1456.0]# sigma_DT, m_DT, theta_DT, T
K_0 = 1.0# Power law constant
index = 1.5# Power slope index
k = 1.0# Noise boost factor
# Saves the universal parameters for the dusty torus, power law and noise boost factor
np.savetxt('Universalparameters.txt',np.concatenate([pDT,[K_0],[index],[k]]),delimiter=',')
    
Jwav = 1250.0 # Wavelength in nm 
JpAD = [1.80,14.62,1.0]# sigma_AD, m_AD, theta_AD
Jnoisescale = 0.1/k
Jband = createdata(f_true, tau, Jwav, K_0, index, pDT, JpAD, Jnoisescale)
np.savetxt('Jband.txt',np.c_[Jband[0],Jband[1]],delimiter=',')
    
Hwav = 1625.0
HpAD = [2.17,16.61,3.0]
Hnoisescale = 0.1/k
Hband = createdata(f_true, tau, Hwav, K_0, index, pDT, HpAD,Hnoisescale)
np.savetxt('Hband.txt',np.c_[Hband[0],Hband[1]],delimiter=',')
    
Kwav = 2150.0
KpAD = [2.23,18.66,5.0]
Knoisescale = 0.1/k
Kband = createdata(f_true, tau, Kwav, K_0, index, pDT, KpAD, Knoisescale)
np.savetxt('Kband.txt',np.c_[Kband[0],Kband[1]],delimiter=',')
    
gwav = 475.4
gpAD = [1.1,11.68,-5.0]
gnoisescale = 0.1/k
gband = createdata(f_true, tau, gwav, K_0, index, pDT, gpAD,gnoisescale)
np.savetxt('gband.txt',np.c_[gband[0],gband[1]],delimiter=',')
    
rwav = 620.4
rpAD = [1.31,12.34,-3.0]
rnoisescale = 0.1/k
rband = createdata(f_true, tau, rwav, K_0, index, pDT, rpAD, rnoisescale)
np.savetxt('rband.txt',np.c_[rband[0],rband[1]],delimiter=',')
    
iwav = 769.8
ipAD = [1.18,13.13,-1.0]
inoisescale = 0.1/k
iband = createdata(f_true, tau, iwav, K_0, index, pDT, ipAD, inoisescale)
np.savetxt('iband.txt',np.c_[iband[0],iband[1]],delimiter=',')
    
zwav = 966.5
zpAD = [1.49,12.55,0.0]
znoisescale = 0.1/k
zband = createdata(f_true, tau, zwav, K_0, index, pDT, zpAD, znoisescale)
np.savetxt('zband.txt',np.c_[zband[0],zband[1]],delimiter=',')
    
# Saves all the accretion disk paramters
np.savetxt('ADparameters.txt',np.c_[JpAD,HpAD,KpAD,gpAD,rpAD,ipAD,zpAD],delimiter=',')
print('Data is saved under "*band.txt" where * is the band. They are ordered as follow: [*-band delta mag, *-band delta mag error]')
print('Dusty torus paramters are saved under "Universalparameters.txt".')
print('Accretion disk paramters are saved under "ADparameters.txt".')


# In[9]:


#Select points at random for all bands to create reduced data
n_list = [86,86,86,86,86,86,86]#number of selected points from all bands J,H,K,g,r,i,z
the_list = list(range(len(X)))
Jind = random.sample(the_list, n_list[0])
Hind = random.sample(the_list, n_list[1])
Kind = random.sample(the_list, n_list[2])
gind = random.sample(the_list, n_list[3])
rind = random.sample(the_list, n_list[4])
iind = random.sample(the_list, n_list[5])
zind = random.sample(the_list, n_list[6])
    
JredX = X[Jind]
redJband = [Jband[0][Jind],Jband[1][Jind]]
np.savetxt('redJband.txt',np.c_[JredX,redJband[0],redJband[1]],delimiter=',')
    
HredX = X[Hind]
redHband = [Hband[0][Hind],Hband[1][Hind]]
np.savetxt('redHband.txt',np.c_[HredX,redHband[0],redHband[1]],delimiter=',')
    
KredX = X[Kind]
redKband = [Kband[0][Kind],Kband[1][Kind]]
np.savetxt('redKband.txt',np.c_[KredX,redKband[0],redKband[1]],delimiter=',')
    
gredX = X[gind]
redgband = [gband[0][gind],gband[1][gind]]
np.savetxt('redgband.txt',np.c_[gredX,redgband[0],redgband[1]],delimiter=',')
    
rredX = X[rind]
redrband = [rband[0][rind],rband[1][rind]]
np.savetxt('redrband.txt',np.c_[rredX,redrband[0],redrband[1]],delimiter=',')
    
iredX = X[iind]
rediband = [iband[0][iind],iband[1][iind]]
np.savetxt('rediband.txt',np.c_[iredX,rediband[0],rediband[1]],delimiter=',')
    
zredX = X[zind]
redzband = [zband[0][zind],zband[1][zind]]
np.savetxt('redzband.txt',np.c_[zredX,redzband[0],redzband[1]],delimiter=',')
    

print('Reduced data is saved as "red*band.txt" where * is the band. data is ordered as follow: [time, deltamag, deltamag_err]')


# In[38]:


#TO DO
#save data as one file

