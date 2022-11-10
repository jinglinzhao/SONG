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

plt.rcParams.update({'font.size': 18})

import jax
import jax.numpy as jnp
from tinygp import kernels, GaussianProcess
import tinygp

jax.config.update("jax_enable_x64", True)

uHz_conv = 1e-6 * 60

amp0, mu0, sig0 = np.array([ 5.46875972e-03,  3.08082489e+03, 3.05370933e+02])
amp0 *= 6e1
nu_max0 = mu0
delta_nu0 = 135
epsilon0 = 0
Q0 = 4e2
N0 = 25 # 24*135/2=1620

def gaussian_2(x, amp, mu, sig):
    return amp * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def generate_acf(nu_max, delta_nu, epsilon, Q, amp, sig, N, filename):

    x = np.linspace(0, 700, 3000)

    kernel = 0

    for i in range(int(N)):
        omega = 2*np.pi*(nu_max + (-(N-1)/2+i)*delta_nu/2 + epsilon)*uHz_conv
        sigma = gaussian_2(nu_max + (-(N-1)/2+i)*delta_nu/2 + epsilon, amp, nu_max, sig)
        kernel += tinygp.kernels.quasisep.SHO(omega, Q, sigma)
        # print(2*np.pi/omega)
        # print(sigma)

    # gp = GaussianProcess(kernel, X, diag=jnp.float64(diag))
    gp = GaussianProcess(kernel, x)


    Nr = 500
    y = gp.sample(jax.random.PRNGKey(4), shape=(Nr,))
    # for i in range(4):
    #     fig = plt.figure(figsize=(16, 2))
    #     idx_x = ((x>=i*200) & (x<(i+1)*200))
    #     plt.plot(x[idx_x], y.T[idx_x], color="k", lw=0.5)
    #     plt.ylabel('RV [m/s]')
    #     plt.xlim([i*200, (i+1)*200])
    #     # plt.ylim([-3.5, 3.5])
    #     plt.show()


    ACF = np.zeros((Nr, len(x)))
    for i in range(Nr):
        ACF[i,:] = sm.tsa.acovf(y[i,:], fft=True)/(1-x/max(x*1.001))


    fig = plt.figure(figsize=(16, 6))
    # plt.plot(x, ACF.T, 'k', alpha=0.05)
    plt.plot(x, np.mean(ACF, axis=0), lw=2, label='mean')
    plt.plot(x, np.median(ACF, axis=0), lw=2, alpha=0.9, label='median')
    plt.title('ACF for simulated data')
    plt.xlim([0,600])
    # plt.ylim([-2,2.0])
    plt.xlabel(r'$\Delta t$ [minutes]')
    plt.ylabel(r'Covariance [m$^2$/s$^2$]')
    plt.legend()

    textstr = '\n'.join((
        r'$\nu_{max}=%.2f$' % nu_max,
        r'$\Delta\nu=%.2f$' % delta_nu,
        r'$\epsilon=%.2f$' % epsilon,
        r'$Q$=%d' % Q,
        r'$S_{0,0}=%.4f$' % amp,
        r'$\sigma=%.4f$' % sig,
        r'$N=%d$' % N
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(350, -0.1, textstr, fontsize=16,
            verticalalignment='top', bbox=props)
    plt.savefig(filename)
    plt.close()