"""A python package to compute derivatives of noisy data using Total Variation Regularization
..moduleauthor:: Swarnavo Sarkar <swarnavo.sarkar@nist.gov>
"""

import os, sys
import numpy as np
import math
import string
from scipy.optimize import brentq, bisect, fixed_point, ridder, newton
import scipy.optimize as opt
import scipy.fftpack as fftpack
import scipy as sp
import scipy.signal as signal
from pylab import *
from numpy.fft import rfft

class RegularizedDifferentiator(object):
    def __init__(self,_x_values):
	"""This constructor takes the discrete domain on which the noisy data is available
	and constructs the coefficient matrices required for a regularization problem."""
	
	self.data_size = len(_x_values)
	self.x_values = _x_values
	
	self.construct_matrices()
	
    def set_regularization_type(self,_regularization_type):
	"""Parameter to decide the type of regularization
	TV - Total Variation (Allows for capturing jumps)
	H1 - H1 regularization (smooths out discontinuities)
	"""
	
	self.regularization_type = _regularization_type

    def construct_matrices(self):
	D = np.matrix(np.zeros(shape=(self.data_size-2,self.data_size-1)))
	#self.step_matrix = np.matrix(np.zeros(shape=(data_size-2,data_size-2)))
	
	for i in range(0,self.data_size-2):
	    for j in range(0,self.data_size-1):
		if i==j:
		    D[i,j] = -1.0/(self.x_values[i+1]-self.x_values[i])
		    #self.step_matrix[i,j] = time_values[i+1]-time_values[i]
		elif j==i+1:
		    D[i,j] = 1.0/(self.x_values[i+1]-self.x_values[i])
	
	self.E_n_sparse = sp.sparse.identity(self.data_size-2,format='csr')
	
	for i in range(0,self.data_size-2):
	    self.E_n_sparse[i,i] = 1.0

	self.D_sparse = sp.sparse.csr_matrix(D)
	
	A = np.matrix(np.zeros(shape=(self.data_size-1,self.data_size-1)))
	B = np.matrix(np.zeros(shape=(self.data_size-1,self.data_size-1)))
	
	for i in range(0,self.data_size-1):
	    for j in range(0,self.data_size-1):
		if i==j:
		    B[i,j] = self.x_values[i+1]-self.x_values[i]

		if j<=i:
		    A[i,j] = self.x_values[j+1]-self.x_values[j]

	mat_1 = A.T*A
	
	self.hess_1 = sp.sparse.csr_matrix(mat_1)
	self.A = sp.sparse.csr_matrix(A)
	self.A_T = sp.sparse.csr_matrix(A.T)
	
    def set_initial_guess(self,_guess):
	if _guess==None:
	    self.initial_guess = np.matrix(np.zeros(shape=(self.data_size-1))).T
	else:
	    self.initial_guess = np.matrix(_guess).T
	    
    def set_tolerance(self,_error_tol):
	self.error_tol = _error_tol
	
    def compute_derivative(self,f,alpha):
	mean_f = 1.0 #np.mean(f)
	size = float(len(f))
	
	f = (1.0/mean_f)*np.array(f)
	
	f_mat = np.matrix(f).T
	#u = np.matrix(np.zeros(shape=(self.data_size-1))).T
	u = np.matrix(self.initial_guess)
	error_norm = 1.0
	
	while error_norm>self.error_tol:
	    s_n = self.compute_update(u,f_mat,alpha)

	    for i in range(0,self.data_size-1):
		u[i,0] += s_n[i]
		
	    error_norm = np.linalg.norm(s_n,2)/np.linalg.norm(u,2)

	diff = self.A*u - f_mat
	
	diff_error = 0.5*diff.T*diff
	
	u = mean_f*u
	
	u_int = self.A*u

	u_list = [u[i,0] for i in range(0,u.shape[0])]
	
	u_int_list = [u_int[i,0] for i in range(0,u_int.shape[0])]
	
	error = (1.0/math.sqrt(float(self.data_size)))*np.linalg.norm(diff,2)
	
	return u_list, u_int_list, error

    def compute_update(self,u,f,alpha): 
	if self.regularization_type=='TV':
	    for i in range(0,self.data_size-2):
		self.E_n_sparse[i,i] = (self.x_values[i+1]-self.x_values[i])/math.sqrt((u[i+1,0]-u[i,0])**2 + 1E-6)
	    
	# Hessian matrix
	hess_2 = self.D_sparse.transpose()*self.E_n_sparse*self.D_sparse
	H_n = self.hess_1 + alpha*hess_2
	
	# Residual
	g_n = -self.hess_1*u + self.A_T*f - alpha*hess_2*u
	
	s_n = sp.sparse.linalg.spsolve(H_n,sp.sparse.csr_matrix(g_n))
	#s_n = sp.sparse.linalg.bicgstab(H_n,g_n)[0]
	
	return s_n

    def root_function(self,alpha):
	self.smooth_rate, self.smooth_set, self.error = self.compute_smooth_derivatives(self.data_set,alpha)
	
	return (self.root_value - self.error)

    def write_filtered_set(self):
	ofile = open(self.testname+self.casename+'_filtered_set.csv','w')
	
	for idx in range(0,len(self.DC_set)):
	    print >> ofile, self.time_set[idx],',',self.DC_set[idx],',',self.temp_set[idx],',',self.dx_set[idx],',',self.sigma_set[idx]
	    
	ofile.close()

    def compute_alpha(self,noise):
	return noise**(0.5)

    def compute_H1_derivative(self,_data,noise):
	"""This function computes the regularized derivative where the 2nd derivative is square integrable
	"""
	self.set_regularization_type('H1')
	self.data_set = _data
	self.root_value = math.sqrt(noise)
	b = self.root_value
	check = False
	
	self.set_initial_guess(None)
	
	while check==False:
	    try:
		sol_alpha = bisect(self.root_function,0.0,b,xtol=1E-6)
		check = True
	    except ValueError:
		b *= 10.0
	
	self.set_initial_guess(self.smooth_set)
	
	alpha = math.sqrt(sol_alpha/self.x_values[-1])
	
	self.set_regularization_type('TV')

	u_list, u_int_list, error = self.compute_derivative(self.data_set,alpha)
	
	return u_list, u_int_list, error

    def root_function(self,alpha):
	self.smooth_rate, self.smooth_set, self.error = self.compute_derivative(self.data_set,alpha)
	
	return (self.root_value - self.error)

    def compute_H1_derivatives(self,raw_t,raw_DC,raw_sigma,raw_temp,raw_dx):
	time_idx = 0
	self.regularization_type = 'Continuous'
	self.noise = {}
	self.alphas = {}
	
	for idx in range(0,len(raw_t)):
	    if raw_t[idx]<=self.calc_time:
		time_idx = idx
	
	self.time_set = list(raw_t[self.negative_data_idx:time_idx])
	self.x_values = self.time_set
	self.construct_normal_matrix(self.time_set)
	
	ofile = open(self.testname+self.casename+'_time_derivatives.csv','w')
	
	self.error_tol = 1e-3
	
	self.noise['DC'] = self.estimate_noise(raw_DC,raw_t)
	self.data_set = raw_DC[self.negative_data_idx+1:time_idx]
	
	self.root_value = math.sqrt(self.noise['DC'])
	b = self.root_value
	check = False
	while check==False:
	    try:
		sol_alpha = bisect(self.root_function,0.0,b,xtol=1E-6)
		check = True
	    except ValueError:
		b *= 10.0
		
	self.DC_p, self.DC_set = self.smooth_rate, self.smooth_set
	print 'alpha, error, root: ', sol_alpha, self.error, self.root_value
	sys.stdout.flush()
	self.alphas['DC'] = sol_alpha
	
	self.noise['sigma'] = self.estimate_noise(raw_sigma,raw_t)
	self.data_set = raw_sigma[self.negative_data_idx+1:time_idx]
	
	self.root_value = math.sqrt(self.noise['sigma'])
	b = self.root_value
	check = False
	while check==False:
	    try:
		sol_alpha = bisect(self.root_function,0.0,b,xtol=1E-10)
		check = True
	    except ValueError:
		b *= 10.0
		
	self.sigma_p, self.sigma_set = self.smooth_rate, self.smooth_set
	print 'alpha, error, root: ', sol_alpha, self.error, self.root_value
	sys.stdout.flush()
	self.alphas['sigma'] = sol_alpha
	
	self.noise['temp'] = self.estimate_noise(raw_temp,raw_t)
	self.data_set = raw_temp[self.negative_data_idx+1:time_idx]
	
	self.root_value = math.sqrt(self.noise['temp'])
	b = self.root_value
	check = False
	while check==False:
	    try:
		sol_alpha = bisect(self.root_function,0.0,b,xtol=1E-6)
		check = True
	    except ValueError:
		b *= 10.0
		
	self.temp_p, self.temp_set = self.smooth_rate, self.smooth_set
	print 'alpha, error, root: ', sol_alpha, self.error, self.root_value
	sys.stdout.flush()
	self.alphas['temp'] = sol_alpha
	
	self.noise['dx'] = self.estimate_noise(raw_dx,raw_t)
	self.data_set = raw_dx[self.negative_data_idx+1:time_idx]
	
	self.root_value = math.sqrt(self.noise['dx'])
	b = self.root_value
	check = False
	while check==False:
	    try:
		sol_alpha = bisect(self.root_function,0.0,b,xtol=1E-10)
		check = True
	    except ValueError:
		b *= 10.0
		
	self.dx_p, self.dx_set = self.smooth_rate, self.smooth_set
	print 'alpha, error, root: ', sol_alpha, self.error, self.root_value
	sys.stdout.flush()
	self.alphas['dx'] = sol_alpha
	
	for idx in range(0,len(self.dx_p)):
	    out_string = str(raw_t[idx])+','+str(self.DC_p[idx])+','+str(self.sigma_p[idx])+','+str(self.temp_p[idx])+','+str(self.dx_p[idx])
	    
	    print >> ofile, out_string
	    
	ofile.close()
	
	self.write_filtered_set()
	
	self.negative_data_idx = 0
	
	print self.alphas
	
    def compute_freq_bounds(self,times):
	min_dt = times[-1]
	max_dt = times[-1]
	
	# Calculate frequency range
	for t_idx in range(0,len(times)-1):
	    dt = times[t_idx+1] - times[t_idx]
	    if min_dt>dt:
		min_dt = dt
	
	# Frequency limits
	max_freq = 2.0*math.pi/min_dt
	min_freq = 2.0*math.pi/max_dt
	
	return max_freq, min_freq
	
    def construct_lowpass_filter(self,times,factor):
	high, low = self.compute_freq_bounds(times)
	# Construct high pass filter
	n = 5
	#high_filter = signal.firwin(n,cutoff=0.05, window = "hamming")
	self.filter_b, self.filter_a = signal.butter(n,factor,'low',analog=False)
	# Spectral inversion
	#high_filter = -high_filter
	#high_filter[n/2] = high_filter[n/2] + 1

    def construct_highpass_filter(self,times,factor):
	high, low = self.compute_freq_bounds(times)
	# Construct high pass filter
	n = 5
	#high_filter = signal.firwin(n,cutoff=0.05, window = "hamming")
	self.filter_b, self.filter_a = signal.butter(n,factor*low,'high',analog=False)

    def filter_this_data(self,data,cutoff_idx):
	imp_ff = signal.filtfilt(self.filter_b, self.filter_a, data)
	
	filtered_set = imp_ff
	#imp_ff = signal.lfilter(self.filter_b, self.filter_a, data)
	#plot(data, color='silver', label='Original')
	#plot(filtered_set, color='#3465a4', label='filtfilt')
	#
	#ofile = open('text.csv','w')
	#
	#for idx in range(0,len(data)):
	#    print >> ofile, data[idx],',',filtered_set[idx]
	#
	#ofile.close()

	#print 'This is the filter: ', signal.convolve(data,high_filter)
	
	#plot(data)
	show()
	
	return filtered_set

    def integrate_rates(self,deriv):
	deriv_mat = np.matrix(deriv).T
	
	integ_set = self.A*deriv_mat
	
    def compute_filter_start_idx(self,data,noise,SNR_threshold):
	for idx in range(25,len(data)-25):
	    SNR = np.std(data[idx-25:idx+25])/math.sqrt(noise)
	    
	    if SNR<SNR_threshold:
		cutoff_idx = idx
		break

	return  cutoff_idx

    def compute_FFT(self,data):
	data_f = fftpack.fft(data)
	
	#plot(data_f)
	#show()
	
    def compute_TV_function(self,_data,_alpha):
	self.data = _data
	self.alpha = _alpha
	u = np.zeros(len(_data))
	
	res = opt.minimize(self.TVobjFun,u)
	
	return list(res.x)
    
    def TVobjFun(self,u):
	total = 0.0
	
	for idx in xrange(0,len(u)):
	    total += 0.5*abs(u[idx]-self.data[idx])**2
	    
	    if idx>0:
		total += self.alpha*abs(u[idx] - u[idx-1])
		
	return total

    def only_regularize(self,data,_alpha):
	self.alpha = _alpha
	n = len(data)
	self.construct_simple_matrices(n)
	
	size = float(len(data))
	
	f = np.array(data)
	
	f_mat = np.matrix(f).T
	#u = np.matrix(np.zeros(shape=(self.data_size-1))).T
	u = np.matrix(self.initial_guess)
	error_norm = 1.0
	
	while error_norm>self.error_tol:
	    s_n = self.compute_simple_update(u,f_mat,self.alpha,n)

	    for i in range(0,n):
		u[i,0] += s_n[i]
		
	    error_norm = np.linalg.norm(s_n,2)/np.linalg.norm(u,2)
	    
	    diff = u - f_mat
	    
	    #error = (1.0/math.sqrt(float(n)))*np.linalg.norm(diff,2)
	    
	    error = np.std(diff)
	    #error_norm = max(abs(s_n))
	    print 'Error:', error_norm, error
	    sys.stdout.flush()

	diff = u - f_mat
	
	diff_error = 0.5*diff.T*diff

	u_list = [u[i,0] for i in range(0,u.shape[0])]
	
	error = (1.0/math.sqrt(float(n)))*np.linalg.norm(diff,2)
	
	return u_list #, error
	
    def construct_simple_matrices(self,n):
	D = np.matrix(np.zeros(shape=(n-1,n)))
	#self.step_matrix = np.matrix(np.zeros(shape=(data_size-2,data_size-2)))
	
	for i in range(0,n-1):
	    for j in range(0,n):
		if i==j:
		    D[i,j] = -1.0
		    #self.step_matrix[i,j] = time_values[i+1]-time_values[i]
		elif j==i+1:
		    D[i,j] = 1.0
	
	self.E_n_sparse = sp.sparse.identity(n-1,format='csr')
	
	for i in range(0,n-1):
	    self.E_n_sparse[i,i] = 1.0

	self.D_sparse = sp.sparse.csr_matrix(D)
	
	self.hess_1 = sp.sparse.identity(n,format='csr')

    def compute_simple_update(self,u,f,alpha,n):
	if self.regularization_type=='TV':
	    for i in range(0,n-1):
		self.E_n_sparse[i,i] = 1.0/math.sqrt((u[i+1,0]-u[i,0])**2 + 1E-6)
	    
	# Hessian matrix
	hess_2 = self.D_sparse.transpose()*self.E_n_sparse*self.D_sparse
	H_n = self.hess_1 + alpha*hess_2
	
	# Residual
	g_n = -u + f - alpha*hess_2*u
	
	s_n = sp.sparse.linalg.spsolve(H_n,sp.sparse.csr_matrix(g_n))
	#s_n = sp.sparse.linalg.bicgstab(H_n,g_n)[0]
	
	return s_n

