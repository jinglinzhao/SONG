from GP_recovery import *
import sys

task_id = int(sys.argv[1])
i = task_id

Nr = 20
t = np.linspace(0, 2000, 2001)
# t = np.linspace(0, 700, 701)
gp = GaussianProcess(kernel, t)
y = gp.sample(jax.random.PRNGKey(4), shape=(Nr,))
# y = y[i,:]
t.shape, y.shape

ACF_stack = np.array([]).reshape(0,2001)
import statsmodels.api as sm

plt.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(16, 6))
ACF = sm.tsa.acovf(y[i,:], fft=True)
# ACF_stack = np.vstack((ACF_stack, ACF))
plt.plot(t, ACF, 'r', lw=2, label='sample')
# plt.plot(t, true_cov, lw=3, color='b', alpha=0.5, label='true kernel')
plt.xlabel(r'$\Delta t$ [minutes]')
plt.ylabel(r'Covariance [m$^2$/s$^2$]')
plt.xlim([0,550])
plt.legend()
plt.savefig('./Figure/GP_recovery_run-19_covariance/' + str(i) + '.png')