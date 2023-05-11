import os
from multiprocessing import Pool

import numpy as np
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
N = 15 # 25*135/2=1620
    
kernel = 0  
for i in range(N):
    omega = 2*np.pi*(nu_max + (-(N-1)/2+i)*delta_nu/2 + epsilon)*uHz_conv
    sigma = gaussian_2(nu_max + (-(N-1)/2+i)*delta_nu/2 + epsilon, amp, nu_max, sig) 
    kernel += tinygp.kernels.quasisep.SHO(omega, Q, sigma)

Nr = 1
t = np.linspace(0, 600, 601)
gp = GaussianProcess(kernel, t)
y = gp.sample(jax.random.PRNGKey(4), shape=(Nr,))
y = y[0,:]
    
def compute_rms(k, t, y, kernel):

    idx_train = (t < 600-20*k) 
    t_train = t[idx_train]
    y_train = y[idx_train]

    gp = GaussianProcess(kernel, t_train)
    cond_gp = gp.condition(y_train, t).gp
    pred_mean, pred_std = cond_gp.loc, cond_gp.variance**0.5   
    
    return np.std(pred_mean[~idx_train] - y[~idx_train])

# if __name__ == '__main__':
#     k = range(4)
#     with Pool(processes=4) as pool:
#         results = pool.imap(compute_rms, zip(k, [t]*4, [y]*4, [kernel]*4))
#         ordered_results = list(results)
#         print(ordered_results)

if __name__ == '__main__':
    num_cores = int(os.getenv('SLURM_CPUS_PER_TASK'))
    k = range(4)
    with Pool(num_cores) as pool:
        results = pool.map(compute_rms, [(k_i, t, y, kernel) for k_i in k])
        print(results)
        

        
# def f(x):
#   return x*x

# if __name__ == '__main__':
#   num_cores = int(os.getenv('SLURM_CPUS_PER_TASK'))
#   with Pool(num_cores) as p:
#     print(p.map(f, [1, 2, 3, 4, 5, 6, 7, 8]))        