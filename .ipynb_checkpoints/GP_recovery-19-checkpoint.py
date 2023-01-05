import numpy as np
import matplotlib.pyplot as plt
from PyAstronomy import pyasl
from datetime import datetime
import celerite
from celerite import terms
from scipy.optimize import minimize
from scipy.signal import find_peaks
from astropy.timeseries import LombScargle
from datetime import timedelta, date
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

from NEID_solar_functions import *
from GP_kernels import *

import jax
import jax.numpy as jnp
from tinygp import kernels, GaussianProcess
import tinygp
import jaxopt

jax.config.update("jax_enable_x64", True)

uHz_conv = 1e-6 * 60

def gaussian_2(x, amp, mu, sig):
    return amp * jnp.exp(-jnp.power(x - mu, 2.) / (2 * jnp.power(sig, 2.))) 

amp, mu, sig = np.array([ 5.46875972e-03,  3.08082489e+03, 3.05370933e+02])    
amp *= 6e1
nu_max = mu
delta_nu = 135
epsilon = 0
Q = 4e2
N = 19 # 25*135/2=1620
    
kernel = 0  
for i in range(N):
    omega = 2*np.pi*(nu_max + (-(N-1)/2+i)*delta_nu/2 + epsilon)*uHz_conv
    sigma = gaussian_2(nu_max + (-(N-1)/2+i)*delta_nu/2 + epsilon, amp, nu_max, sig) 
    kernel += tinygp.kernels.quasisep.SHO(omega, Q, sigma)
    print(sigma)
