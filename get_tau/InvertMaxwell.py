#!/usr/bin/env python

import os, sys
import numpy as np
import math
import string
from scipy.optimize import brentq, bisect, fixed_point, ridder, newton
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib import rc
import scipy as sp
import scipy.signal as signal
from RegularizedDifferentiator import RegularizedDifferentiator
import scipy.optimize as optimizer

class InvertMaxwell(object):
    def __init__(self,input_lines):
	"""This function initializes the data structures for calculation and options for displaying figures.
	
	Args::
	    input_lines (str): The string to use.
	    
	    show_plot (bool): Boolean to decide display of figures
	"""
	
	input_dict = {}
	
	for l in input_lines:
	    if string.find(l,' = ') != -1:
		thisSet = l.split(' = ')
		thisSet[0] = thisSet[0]
		thisSet[1] = thisSet[1].rstrip('\r\n')
		input_dict[thisSet[0]] = thisSet[1]

	self.super_directory = input_dict['Superfolder']
	
	self.sub_directory = input_dict['Subfolder']
	
	self.testname = input_dict['Testname']
	
	self.origin_directory = os.getcwd()
	
	os.chdir('./'+self.super_directory+'/'+self.sub_directory)
	
	self.get_sample_info()

        self.eta_0 = float(input_dict['Initial viscosity (in Pa.s)'])
	self.E = (1E+9)*float(input_dict['High-frequency Youngs modulus (GPa)'])

        self.calc_time = float(input_dict['Calculation time (s)'])
	self.max_chem_strain = float(input_dict['Maximum shrinkage strain'])

	thermal_coeff_string = input_dict['Coefficient of thermal expansion (in 1/T)']
		
	coeff_set = thermal_coeff_string.split(',')
	
	self.monomer_alpha = float(coeff_set[0])
	self.polymer_alpha = float(coeff_set[1])
	
	self.rho_cl_max = float(input_dict['Molecular density (in moles/m^3)'])
	
	self.regularization_parameter = float(input_dict['Regularization parameter'])
	
	self.denoising_window = 2*int(input_dict['Denoising window'])+1
	
	self.penalty_parameter = 0.1
	    
	self.casename = input_dict['Casename']
	
	# Area and multiplier to compute stress from displacement
	self.sample_area = 0.25*math.pi*(self.sample_d**2)
	
	self.stress_factor = 1.0/(self.compliance*self.sample_area)
	
	self.read_data()
	
	# Variables for time integration
	self.current_time = 0.0
	self.t_step = 0.0

    def __del__(self):
	os.chdir(self.origin_directory)
	
    def get_sample_info(self):
	"""This function reads the sample geometry, beam compliance, and irradiation duration.
	"""
	
	infile = open(self.testname+'.txt','r')
	
	all_lines = infile.readlines()
	
	infile.close()
	
	sample_dict = {}
	
	for line in all_lines:
	    if string.find(line,':') != -1:
		thisSet = line.split(':')
		thisSet[0] = thisSet[0]
		thisSet[1] = thisSet[1].rstrip('\r\n')
		sample_dict[thisSet[0]] = thisSet[1].lstrip()

	self.sample_x = 0.01*float(sample_dict['Sample height (cm)'])
	self.sample_d = 0.01*float(sample_dict['Diamater of sample (cm)'])
	self.compliance = (1E-2)*float(sample_dict['Compliance of the beam (cm/N)'])
	self.curing_time = float(sample_dict['Duration of irradiation (s)'])

	self.sample_position = float(sample_dict['Position of sample along the beam (cm)'])
	
	self.displacement_factor = 1.0/(3*(24.5 - self.sample_position)/(2*self.sample_position) + 1.0)

    def read_data(self):
	"""This function reads the time evolution data for the following 4 quantities.
	DC - Degree of Conversion
	T - Temperature
	$\sigma$ - Stress
	$\Delta h$ - Displacement
	"""
	
	self.time_t = []
	self.DC_t = []
	self.sigma_t = []
	self.temp_t = []
	self.dx_t = []
	
	raw_t, raw_DC, raw_sigma, raw_temp, raw_dx = [], [], [], [], []
	
	#rfile = open(self.testname+'clean_data.txt','r')
	rfile = open(self.testname+'Clean.csv','r')
	
	all_lines = rfile.readlines()
	
	# Get index of tags from the header
	head_set = all_lines[0].rstrip('\r\n').split(',')
	
	time_idx, stress_idx, DC_idx, dx_idx, temp_idx = None, None, None, None, None
	
	for tag_idx in range(0,len(head_set)):
	    if 'Time' in head_set[tag_idx]:
		time_idx = tag_idx
	    elif 'DC' in head_set[tag_idx]:
		DC_idx = tag_idx
	    elif 'Stress' in head_set[tag_idx]:
		stress_idx = tag_idx
	    elif 'Disp.' in head_set[tag_idx]:
		dx_idx = tag_idx
	    elif 'deltaT' in head_set[tag_idx]:
		temp_idx = tag_idx
	
	if time_idx==None:
	    print 'Time information missing. Viscosity calculation is impossible.'
	    sys.stdout.flush()
	    sys.exit()
	if DC_idx==None:
	    print 'Degree of Conversion missing. Viscosity calculation is impossible.'
	    sys.stdout.flush()
	    sys.exit()
	if dx_idx==None:
	    print 'Displacement values missing. Viscosity calculation is impossible.'
	    sys.stdout.flush()
	    sys.exit()
	if stress_idx==None:
	    print 'Stress values missing. Viscosity calculation is impossible.'
	    sys.stdout.flush()
	    sys.exit()
	if temp_idx==None:
	    print 'Temperature values missing. Thermal strain is neglected.'
	    
	for line in all_lines[1:]:
	    this_set = line.rstrip('\r\n').split(',')
	    time = float(this_set[0])
	    if time>=0.0:
		raw_t.append(float(this_set[time_idx]))
		raw_sigma.append(float(this_set[stress_idx]))
		raw_dx.append(self.displacement_factor*float(this_set[dx_idx]))
		raw_DC.append(float(this_set[DC_idx]))
		
		try:
		    raw_temp.append(float(this_set[temp_idx]))
		except TypeError:
		    raw_temp.append(0.0)
		
	self.data_size = len(raw_t)
	
	self.negative_data_idx = 0
	
	self.lights_off_idx = 0
	
	for idx in range(0,self.data_size):
	    if raw_t[idx]<self.curing_time:
		if raw_sigma[idx]<0.0 and raw_DC[idx]<40.0:
		    self.negative_data_idx = idx
		
	for idx in range(0,self.negative_data_idx):
	    raw_DC[idx], raw_sigma[idx],raw_temp[idx], raw_dx[idx] = 0.0, 0.0, 0.0, 0.0
	
	try:
	    os.mkdir(self.testname+self.casename+'_outputs')
	except OSError:
	    pass
	
	os.chdir(self.testname+self.casename+'_outputs')
	    
	#self.compute_H1_derivatives(raw_t,raw_DC,raw_sigma,raw_temp,raw_dx)
	self.compute_derivatives(raw_t,raw_DC,raw_sigma,raw_temp,raw_dx)
	
	self.compute_low_modulus()
	
	self.max_DC = max(raw_DC)
	
    def compute_eta(self):
	"""This function computes the viscosity value by directly substituting the rate of the observables
	and the high-frequency modulus into the governing equation of a Maxwell material.
	"""
	
	self.initialize_computed_set()
	
	self.total_radial_stress = 0.0
	
	#ofile = open(self.testname+self.casename+'_eta.csv','w')
	#print >> ofile, self.evaluated_time[0],',', self.evaluated_DC[0],',', math.log10(self.evaluated_eta[0]),',', ((1E-6)*self.evaluated_sigma[0])
	#print >> ofile, self.time_set[0],',', 0.0,',', math.log10(self.eta_0)
	
	tau_values = []
	time_set_idx = []
	
	numerator = []
	denominator = []
	denom1, denom2 = [], []
	
	d_e_chem_r, d_e_temp_r, d_e_total_r, d_e_mech_r = [], [], [], []
	curing_idx = self.get_curing_idx()
	
	for t_idx in range(0,len(self.time_set)-1):
	    check = False
	    
	    # Time step
	    #dt = 2.0*self.minimum_time_step
	    
	    # Chemical strain update
	    d_e_chem_r.append(-(1.0/3.0)*self.max_chem_strain*self.DC_p[t_idx]/self.max_DC)
	    
	    # Thermal strain update
	    thermal_coeff = self.polymer_alpha*(self.DC_set[t_idx]/self.max_DC) + self.monomer_alpha*(1.0 - self.DC_set[t_idx]/self.max_DC)
	    d_e_temp_r.append((1.0/3.0)*thermal_coeff*self.temp_p[t_idx])
	    
	    # Total strain update
	    d_e_total_r.append(-(1E-3)*self.dx_p[t_idx]/self.sample_x)
	    # Mechanical strain update
	    d_e_mech_r.append(d_e_total_r[-1] - (d_e_chem_r[-1] + d_e_temp_r[-1]))
	    # Observed stress update
	    d_obs_sigma_r = (1E+6)*self.sigma_p[t_idx]
	    # Stress rate
	    stress_rate = d_obs_sigma_r
	    # Strain rate
	    strain_rate = d_e_mech_r[-1]
	    
	    # Chemical strain update
	    e_chem = -(1.0/3.0)*self.max_chem_strain*self.DC_set[t_idx]/self.max_DC
	    # Thermal strain update
	    e_temp = (1.0/3.0)*thermal_coeff*self.temp_set[t_idx]
	    # Total strain
	    e_total = -(1E-3)*self.dx_set[t_idx]/self.sample_x
	    
	    # Mechanical strain
	    strain = e_total - (e_chem + e_temp)
	    
	    # Eta factor
	    denominator.append(self.E*d_e_mech_r[-1] - stress_rate)
	    
	    numerator.append((1E+6)*self.sigma_set[t_idx] - self.E1_set[t_idx]*strain)
	    
	    denom1.append(self.E*d_e_mech_r[-1])
	    
	    denom2.append(stress_rate)
	    
	# Filter the coefficients to tau-function from SLS model
	numerator, denom1, denom2, index_set = self.filter_factors(numerator,denom1,denom2)

	for idx in xrange(0,len(numerator)):
	    if self.time_set[index_set[idx]]>=self.calc_time:
		break
	    elif (denom1[idx]-denom2[idx])!=0.0:
		tau = numerator[idx]/(denom1[idx]-denom2[idx])
		
		if tau>0.0:
		    tau_values.append(math.log10(tau))
		    time_set_idx.append(index_set[idx])
		    
	for idx in time_set_idx:
	    if self.time_set[idx]>self.curing_time:
		cutoff_idx = idx
		break

	tau_values = self.filter_exponent(tau_values,cutoff_idx)
	
	self.DC_set = self.write_filtered_inputs(self.DC_set)
	self.dx_set = self.write_filtered_inputs(self.dx_set)
	self.temp_set = self.write_filtered_inputs(self.temp_set)
	self.sigma_set = self.write_filtered_inputs(self.sigma_set)
	self.E1_set = self.write_filtered_inputs(self.E1_set)

	new_time, new_DC, new_sigma, new_temp, new_dx, new_E1 = [], [], [], [], [], []
	
	for t_idx in time_set_idx:
	    new_time.append(self.time_set[t_idx])
	    new_DC.append(self.DC_set[t_idx])
	    new_sigma.append(self.sigma_set[t_idx])
	    new_temp.append(self.temp_set[t_idx])
	    new_dx.append(self.dx_set[t_idx])
	    new_E1.append(self.E1_set[t_idx])
	    
	    self.eval_d_e_chem.append(d_e_chem_r[t_idx])
	    self.eval_d_e_temp.append(d_e_temp_r[t_idx])
	    self.eval_d_e_total.append(d_e_total_r[t_idx])
	    self.eval_d_e_mech.append(d_e_mech_r[t_idx])
	    
	self.DC_set = new_DC
	self.time_set = new_time
	self.dx_set = new_dx
	self.temp_set = new_temp
	self.sigma_set = new_sigma
	self.E1_set = new_E1
	
	for t_idx in range(0,len(self.time_set)-1):
	    #eta = tau_values[t_idx]*(self.E - self.E1_set[t_idx])
	    tau = 10**tau_values[t_idx]
	    
	    #if tau>0.0:
	    self.evaluated_time.append(self.time_set[t_idx])
	    self.evaluated_DC.append(self.DC_set[t_idx])
	    self.evaluated_sigma.append(self.sigma_set[t_idx])
	    self.evaluated_temp.append(self.temp_set[t_idx])
	    self.evaluated_dx.append(self.dx_set[t_idx])
	    self.evaluated_tau.append(tau)
	    self.evaluated_E1.append(self.E1_set[t_idx])
	    self.evaluated_E2.append(self.E - self.E1_set[t_idx])
	    


	    #print >> ofile, self.evaluated_time[-1],',', self.evaluated_DC[-1],',', self.evaluated_tau[-1]

	#ofile.close()
	
	self.data_size = len(self.evaluated_DC)

    def initialize_computed_set(self):
	"""This function initializes the data structures used to store the observables.
	"""
	
	self.evaluated_time = []
	#self.evaluated_time.append(self.time_set[0])
	
	self.evaluated_DC = []
	#self.evaluated_DC.append(self.DC_set[0])
	
	self.evaluated_tau = []
	#self.evaluated_eta.append(self.eta_0)
	
	self.evaluated_sigma = []
	#self.evaluated_sigma.append(self.sigma_set[0])
	
	self.evaluated_temp = []
	#self.evaluated_temp.append(self.temp_set[0])
	
	self.evaluated_dx = []
	#self.evaluated_dx.append(self.dx_set[0])
	
	# Computed value of low frequency Young's modulus of the soft matter
	self.evaluated_E1 = []
	
	# Computed value of dashpot Young's modulus of the soft matter
	self.evaluated_E2 = []
	
	# Strains
	self.eval_d_e_temp, self.eval_d_e_chem, self.eval_d_e_total, self.eval_d_e_mech = [], [], [], []
	
	# Crosslinks density of the material
	self.evaluated_crosslinks = []
	
    def eliminate(self,sigma,d_obs_sigma):
	check = False
	
	if sigma==0.0:# and d_obs_sigma==0.0:
	    check = True
	#if d_obs_sigma<0.0 and sigma<0.0:
	#    check = True
	
	return check

    def plot_results(self):
	"""This function plots the results.
	"""
	#plt.plot(self.evaluated_time,self.evaluated_tau,marker='o',ms=4,lw=0.0)
	plt.plot(self.evaluated_time,self.smoothed_tau,marker='x',ms=0.5,lw=3.0,mfc='r')
	plt.minorticks_on()
	plt.grid()
	plt.xlim(0.0,max(self.evaluated_time))
	plt.xticks(size=20)
	plt.yticks(size=20)
	plt.xlabel(r'Time, $t$',size=20)
	plt.ylabel(r'Viscosity, $\eta(\mathrm{Pa.s})$',size=20)
	plt.yscale('log')
	plt.show()
	
    def compute_complex_modulus(self):
	"""This function computes the storage and the loss modulus development of the material using the smoothed viscosity trajectory.
	"""
	# Create frequency range
	freq_high = 1E+3 #(2*math.pi)*self.E/self.eta_0
	freq_low = 1E-3 #(2*math.pi)*self.E/self.fitted_eta_values[-1]
	
	freq_set = np.linspace(math.log10(freq_low),math.log10(freq_high),7)
	
	ofile_1 = open(self.testname+self.casename+'_storage_modulus.csv','w')
	
	ofile_2 = open(self.testname+self.casename+'_loss_modulus.csv','w')
	
	outstring = 'DC'+','+'Time'
	
	for log_omega in freq_set:
	    outstring += ','+str(10**log_omega)
	
	print >> ofile_1, outstring
	print >> ofile_2, outstring
	
	for idx in range(0,len(self.smoothed_eta)):
	    #tau = min(self.evaluated_eta[idx],self.fitted_eta_values[idx])/self.E
	    tau = self.smoothed_eta[idx]/self.E
	    #tau = (10**self.smooth_l_eta[idx])/self.E
	    
	    outstring_1 = str(self.evaluated_DC[idx])+','+str(self.evaluated_time[idx])
	    outstring_2 = str(self.evaluated_DC[idx])+','+str(self.evaluated_time[idx])
	    
	    for log_omega in freq_set:
		storage_modulus, loss_modulus = self.get_modulus_values(tau,float(10**log_omega))
		
		outstring_1 += ','+str(storage_modulus)
		outstring_2 += ','+str(loss_modulus)
		
	    print >> ofile_1, outstring_1
	    print >> ofile_2, outstring_2

	ofile_1.close()
	ofile_2.close()
		
    def get_modulus_values(self,tau,omega):
	"""Obtains the two moduli values for a given combination of:
	tau - Relaxation time constant
	omega - loading frequency
	"""
	E_1 = 1.0/(1.0 + 1.0/((tau*omega)**2))
	
	E_2 = 1.0/((tau*omega) + 1.0/(tau*omega))
	
	return ((1E-9)*self.E*E_1), ((1E-9)*self.E*E_2)

    def compute_radial_stress(self,d_e_exp,d_e_mech,d_sigma_obs,t_idx,dt):
	# Size of this time interval
	integrate_step = 0.5*(self.time_set[t_idx]-self.time_set[t_idx-1]) + 0.5*(self.time_set[t_idx+1]-self.time_set[t_idx])
	
	d_e_obs = d_e_exp*self.E*0.5*self.sample_d/(self.E*0.5*self.sample_d + self.tube_modulus*self.tube_thickness)
	#d_e_obs = 0.0
	
	# Radial mechanical strain increment
	d_e_mech_r = d_e_obs - d_e_exp
	
	d_sigma_comp = self.E*(d_e_obs - d_e_exp)
	
	# Deviatoric stress increment
	d_e_sigma = d_sigma_obs - (1.0/3.0)*(2.0*d_sigma_comp + d_sigma_obs)
	
	# Deviatoris strain increment
	d_e_mech = d_e_mech - (1.0/3.0)*(2.0*(d_e_obs - d_e_exp) + d_e_mech)
	
	self.total_radial_stress += d_sigma_comp*(integrate_step/dt)
	
	#print self.total_radial_stress, (1E+6)*self.sigma_set[t_idx]
	
	#print self.time_set[t_idx], d_sigma_obs, d_sigma_comp
	
	#if d_e_sigma/d_e_mech<0.0:
	#    print 'Error: ', self.time_set[t_idx]
	
	return d_e_sigma, d_e_mech

    def compute_eta_exponent(self):
	"""This function computes the derivative of viscosity trajectory
	with respect to Degree of Conversion.
	"""
	
	self.DC_CR = 66.66
	
	#ofile = open(self.testname + self.casename + '_eta_exponent.csv','w')
	
	l_tau, l_dc, l_t = [], [], []
	
	initial_l_tau = math.log10(self.evaluated_tau[0])
	#initial_l_tau = self.evaluated_tau[0]

	for idx in range(0,len(self.evaluated_DC)):
	    l_dc.append(math.log10(self.DC_CR - self.evaluated_DC[idx]))
	    l_tau.append(math.log10(self.evaluated_tau[idx])-initial_l_tau)
	    l_t.append(math.log10(self.evaluated_time[idx]))
	    #l_tau.append(self.evaluated_tau[idx])
	    #l_t.append(self.evaluated_time[idx])
	
	eta_differentiator = RegularizedDifferentiator(l_dc)
	eta_differentiator.set_regularization_type('H1')
	eta_differentiator.set_initial_guess(None)
	eta_differentiator.set_tolerance(1e-6)
	
	noise = self.estimate_noise(l_tau,l_t)
	alpha = self.compute_alpha(noise)
	self.tau_exp, l_tau, self.error = eta_differentiator.compute_derivative(l_tau[1:],self.regularization_parameter)
	
	self.tau_exp.insert(0,0.0)
	
	for idx in xrange(0,len(self.tau_exp)):
	    self.tau_exp[idx] *= -1
	
	self.smooth_l_tau = []
	self.smooth_l_tau.append(initial_l_tau)
	
	for tau in l_tau:
	    self.smooth_l_tau.append(tau+initial_l_tau)
	
	self.smoothed_tau = []
	self.smoothed_eta = []
	
	for eta_idx in range(0,len(self.tau_exp)):
	    self.smoothed_tau.append(10**self.smooth_l_tau[eta_idx])
	    #self.smoothed_tau.append(self.smooth_l_tau[eta_idx])
	    self.smoothed_eta.append(self.smoothed_tau[eta_idx]*self.evaluated_E2[eta_idx])
	    #print >> ofile, self.evaluated_time[eta_idx],',',self.evaluated_DC[eta_idx],',',self.evaluated_tau[eta_idx],',',self.smooth_l_tau[eta_idx],',',self.tau_exp[eta_idx]
	    
	#ofile.close()
	
	#self.smoothed_tau.append(10**self.smooth_l_tau[-1])

    def compute_regularized_derivatives(self,f,f_prime_guess):
	# Compute penalty term
	penalty_value = 0.0
	
	for t_idx in range(0,self.data_size-1):
	    # Second derivative
	    f_prime_prime = f_prime_guess/(0.5*(self.time_t[t_idx+1] - self.time_t[t_idx-1]))
	    
	    penalty_value += abs(f_prime_prime)*(0.5*(self.time_t[t_idx+1] - self.time_t[t_idx-1]))
	    
    def construct_normal_matrix(self,x_values):
	data_size = len(x_values)

	D = np.matrix(np.zeros(shape=(data_size-2,data_size-1)))
	#self.step_matrix = np.matrix(np.zeros(shape=(data_size-2,data_size-2)))
	
	for i in range(0,data_size-2):
	    for j in range(0,data_size-1):
		if i==j:
		    D[i,j] = -1.0/(x_values[i+1]-x_values[i])
		    #self.step_matrix[i,j] = time_values[i+1]-time_values[i]
		elif j==i+1:
		    D[i,j] = 1.0/(x_values[i+1]-x_values[i])
	
	self.E_n_sparse = sp.sparse.identity(data_size-2,format='csr')

	self.D_sparse = sp.sparse.csr_matrix(D)
	
	A = np.matrix(np.zeros(shape=(data_size-1,data_size-1)))
	B = np.matrix(np.zeros(shape=(data_size-1,data_size-1)))
	
	for i in range(0,data_size-1):
	    for j in range(0,data_size-1):
		if i==j:
		    B[i,j] = x_values[i+1]-x_values[i]

		if j<=i:
		    A[i,j] = x_values[j+1]-x_values[j]

	mat_1 = A.T*A
	
	self.hess_1 = sp.sparse.csr_matrix(mat_1)
	self.A = sp.sparse.csr_matrix(A)
	self.A_T = sp.sparse.csr_matrix(A.T)
	
    def compute_smooth_derivatives(self,f,alpha):
	len_f = len(f)
	f_mat = np.matrix(f).T
	u = np.matrix(np.zeros(shape=(len_f))).T
	error_norm = 1.0
	
	while error_norm>self.error_tol:
	    s_n = self.compute_update(u,f_mat,alpha)

	    for i in range(0,len_f):
		u[i,0] += s_n[i]
	    #error_norm = np.linalg.norm(s_n,2)/np.linalg.norm(u,2)
	    error_norm = max(abs(s_n))#np.linalg.norm(s_n,'inf')

	diff = self.A*u - f_mat
	
	diff_error = 0.5*diff.T*diff
	
	u_int = self.A*u

	u_list = [u[i,0] for i in range(0,u.shape[0])]
	
	u_int_list = [u_int[i,0] for i in range(0,u_int.shape[0])]
	
	error = (1.0/math.sqrt(float(len_f)))*np.linalg.norm(diff,2)
	
	print 'Error: ', error, alpha
	sys.stdout.flush()
	
	return u_list, u_int_list, error

    def compute_update(self,u,f,alpha): 
	if self.regularization_type=='Discontinuous':
	    for i in range(0,len(u)-1):
		self.E_n_sparse[i,i] = (self.x_values[i+1]-self.x_values[i])/math.sqrt((u[i+1,0]-u[i,0])**2 + 1E-6)
	    
	# Hessian matrix
	hess_2 = self.D_sparse.transpose()*self.E_n_sparse*self.D_sparse
	#hess_2 = self.E_n_sparse
	H_n = self.hess_1 + alpha*hess_2
	
	# Residual
	g_n = -self.hess_1*u + self.A_T*f - alpha*hess_2*u
	
	s_n = sp.sparse.linalg.spsolve(H_n,sp.sparse.csr_matrix(g_n))
	#s_n = sp.sparse.linalg.cg(H_n,g_n)[0]
	#s_n = sp.sparse.linalg.gmres(H_n,g_n)[0]
	
	return s_n

    def compute_derivatives(self,raw_t,raw_DC,raw_sigma,raw_temp,raw_dx):
	"""This function computes the time derivatives of the observables
	using Total Variation Regularization.
	"""
	
	for idx in range(0,len(raw_t)):
	    if raw_t[idx]<=self.calc_time:
		self.time_idx = idx

	self.time_set = list(raw_t[self.negative_data_idx:])
	
	curing_idx = self.get_curing_idx()
	
	win_size = 0
	
	self.time_differentiator_1 = RegularizedDifferentiator(self.time_set)
	self.time_differentiator_1.set_regularization_type('TV')
	self.time_differentiator_1.set_initial_guess(None)
	self.time_differentiator_1.set_tolerance(1E-6)

	self.noise = {}
	
	self.noise['DC'] = self.estimate_noise(raw_DC,raw_t)
	alpha = self.compute_alpha(self.noise['DC'])
	data_set = raw_DC[self.negative_data_idx+1:]
	
	self.DC_p, self.DC_set = self.get_this_derivative(data_set,curing_idx,0.0,win_size)
	self.DC_set.insert(0,0.0)
	
	self.noise['sigma'] = self.estimate_noise(raw_sigma,raw_t)
	alpha = self.compute_alpha(self.noise['sigma'])
	data_set = raw_sigma[self.negative_data_idx+1:]
	self.sigma_p, self.sigma_set = self.get_this_derivative(data_set,curing_idx,0.0,win_size)
	self.sigma_set.insert(0,0.0)
	
	self.noise['dx'] = self.estimate_noise(raw_dx,raw_t)
	alpha = self.compute_alpha(self.noise['dx'])
	data_set = raw_dx[self.negative_data_idx+1:]
	self.dx_p, self.dx_set = self.get_this_derivative(data_set,curing_idx,0.0,win_size)  
	self.dx_set.insert(0,0.0)
	
	self.noise['temp'] = self.estimate_noise(raw_temp,raw_t)
	alpha = self.compute_alpha(self.noise['temp'])
	data_set = raw_temp[self.negative_data_idx+1:]
	self.temp_p, self.temp_set = self.get_this_derivative(data_set,curing_idx,0.0,win_size)
	self.temp_set.insert(0,0.0)
	
	#new_time_set = self.time_set[:curing_idx-win_size] + self.time_set[curing_idx+1+win_size:]
	#self.time_set = new_time_set
	
	#self.write_time_derivatives()

    def get_this_derivative(self,data,curing_idx,alpha,win_size):
	data_p1, data_1, error = self.time_differentiator_1.compute_derivative(data,alpha)
	
	return data_p1, data_1

    def write_time_derivatives(self):
	ofile = open(self.testname+self.casename+'_regularized_values.csv','w')
	ofile_p = open(self.testname+self.casename+'_time_derivatives.csv','w')
	
	for idx in range(0,len(self.dx_p)):
	    out_string_p = str(self.time_set[idx])+','+str(self.DC_p[idx])+','+str(self.sigma_p[idx])+','+str(self.temp_p[idx])+','+str(self.dx_p[idx])
	    out_string = str(self.time_set[idx])+','+str(self.DC_set[idx])+','+str(self.sigma_set[idx])+','+str(self.temp_set[idx])+','+str(self.dx_set[idx])
	    
	    print >> ofile, out_string
	    print >> ofile_p, out_string_p
	    
	ofile.close()
	ofile_p.close()
	
    def root_function(self,alpha):
	self.smooth_rate, self.smooth_set, self.error = self.compute_smooth_derivatives(self.data_set,alpha)
	
	return (self.root_value - self.error)

    def estimate_noise(self,data,times):
	set_size = 100
	mean_value = np.mean(data[len(data)-set_size:len(data)])
	std_value = np.std(data[len(data)-set_size:len(data)])
	
	noise = std_value#/math.sqrt(float(set_size))
	
	return noise

    def write_filtered_set(self):
	ofile = open(self.testname+self.casename+'_filtered_set.csv','w')
	
	for idx in range(0,len(self.DC_set)):
	    print >> ofile, self.time_set[idx],',',self.DC_set[idx],',',self.temp_set[idx],',',self.dx_set[idx],',',self.sigma_set[idx]
	    
	ofile.close()

    def compute_alpha(self,noise):
	return noise

    def compute_low_modulus(self):
	self.crosslinks_set = []
	self.E1_set = []
	
	for idx in xrange(0,len(self.DC_set)):
	    self.crosslinks_set.append((0.01*self.DC_set[idx])**2)
	    
	    E_1 = 3*8.314*(300+self.temp_set[idx])*self.rho_cl_max
	    
	    self.E1_set.append(E_1*self.crosslinks_set[-1])
	    
    def find_DCthreshold_idx(self,data_set):
	max_temp = max(data_set)

	max_idx = data_set.index(max_temp)
	
	#this_idx = 0
	#
	#for idx in xrange(len(DC_set)):
	#    if threshold<DC_set[idx]:
	#	this_idx = idx
	#	break
		
	return max_idx

    def get_curing_idx(self):
	this_idx = None
	
	for idx in xrange(0,len(self.time_set)):
	    if self.time_set[idx]>self.curing_time:
		this_idx = idx
		break

	return this_idx

    def write_essential_results(self):
	ofile = open(self.testname+self.casename+'_main_outs.csv','w')
	
	print >> ofile, 'Time,DC,Tau,E_1,E_2,d_e_mech'
	
	for idx in xrange(0,len(self.evaluated_time)):
	    outstring = str(self.evaluated_time[idx])+','+str(self.evaluated_DC[idx])+','+str(self.smoothed_tau[idx])+','+str(self.evaluated_E1[idx])+','+str(self.evaluated_E2[idx])
	    outstring += ','+str(self.eval_d_e_mech[idx])
	    
	    print >> ofile, outstring
	    
	ofile.close()
	
    def extract_fit_set(self,data_x,data_y):
	#print data_x, data_y
	noise = (1e-6)*self.estimate_noise(data_y,data_x)
	
	popt, pcov = optimizer.curve_fit(self.func,data_x,data_y,p0=[1,-1,data_x[0]],ftol=noise)
	
	fit_values = []
	
	for x in data_x:
	    fit_values.append(self.func(x,popt[0],popt[1],popt[2]))
	
	return fit_values
	
    def func(self,x,a,b,c):
	f = a*(x**b) + c
	
	return f

    def smooth_derivatives(self):
	temp_idx = self.find_DCthreshold_idx(self.temp_set)
	curing_idx = self.get_curing_idx()
	len_set = len(self.time_set)
	
	DC_set1, DC_set2 = self.DC_set[:temp_idx], self.DC_set[temp_idx:]
	DC_set1 = signal.savgol_filter(DC_set1,11,2,1)
	DC_set2 = signal.savgol_filter(DC_set2,51,2,1)
	
	self.DC_p = []
	
	for DC in DC_set1:
	    self.DC_p.append(DC)
	    
	for DC in DC_set2:
	    self.DC_p.append(DC)

	sigma_set1, sigma_set2 = self.sigma_set[:curing_idx], self.sigma_set[curing_idx:]
	sigma_set1 = signal.savgol_filter(sigma_set1,11,2,1)
	sigma_set2 = signal.savgol_filter(sigma_set2,51,2,1)
	
	self.sigma_p = []
	
	for sigma in sigma_set1:
	    self.sigma_p.append(sigma)
	    
	for sigma in sigma_set2:
	    self.sigma_p.append(sigma)
	
	temp_set1, temp_set2 = self.temp_set[:curing_idx], self.temp_set[curing_idx:]
	temp_set1 = signal.savgol_filter(temp_set1,11,2,1)
	temp_set2 = signal.savgol_filter(temp_set2,51,2,1)
	
	self.temp_p = []
	
	for temp in temp_set1:
	    self.temp_p.append(temp)
	    
	for temp in temp_set2:
	    self.temp_p.append(temp)
	    
	dx_set1, dx_set2 = self.dx_set[:curing_idx], self.dx_set[curing_idx:]
	dx_set1 = signal.savgol_filter(dx_set1,11,2,1)
	dx_set2 = signal.savgol_filter(dx_set2,51,2,1)
	
	self.dx_p = []
	
	for dx in dx_set1:
	    self.dx_p.append(dx)
	    
	for dx in dx_set2:
	    self.dx_p.append(dx)
	    
	sigma_set1, sigma_set2 = self.sigma_set[:curing_idx], self.sigma_set[curing_idx:]
	sigma_set1 = signal.wiener(sigma_set1,10)
	sigma_set2 = signal.savgol_filter(sigma_set2,51,2)
	
	self.sigma_set = []
	
	for sigma in sigma_set1:
	    self.sigma_set.append(sigma)
	    
	for sigma in sigma_set2:
	    self.sigma_set.append(sigma)

    def filter_factors(self,numerator,denom1,denom2):
	curing_idx = self.get_curing_idx()
	
	if curing_idx>500:
	    curing_idx = 500
	
	win1, win2 = self.denoising_window, 5*self.denoising_window
	
	gap = 2*win1
	
	if curing_idx!=None:
	    index_set = range(0,curing_idx-gap) + range(curing_idx+gap,len(numerator))
	    
	    numerator = numerator[:curing_idx-gap] + numerator[curing_idx+gap:]
	    denom1 = denom1[:curing_idx-gap] + denom1[curing_idx+gap:]
	    denom2 = denom2[:curing_idx-gap] + denom2[curing_idx+gap:]
	    
	#num1 = signal.savgol_filter(numerator,win1,1)
	#denom1_1 = signal.savgol_filter(denom1,win1,1)
	#denom2_1 = signal.savgol_filter(denom2,win1,1)
	#
	#num2 = signal.savgol_filter(numerator,win2,1)
	#denom1_2 = signal.savgol_filter(denom1,win2,1)
	#denom2_2 = signal.savgol_filter(denom2,win2,1)
	
	num1 = signal.wiener(numerator,win1)
	denom1_1 = signal.wiener(denom1,win1)
	denom2_1 = signal.wiener(denom2,win1)
	
	num2 = signal.wiener(numerator,win2)
	denom1_2 = signal.wiener(denom1,win2)
	denom2_2 = signal.wiener(denom2,win2)
	
	#num3 = signal.savgol_filter(numerator[curing_idx-win2:],win3,2)
	#denom3 = signal.wiener(denominator[2*curing_idx:],win3)
	
	##num1 = self.extract_fit_set(tset1,numerator[:temp_idx])
	##num2 = self.extract_fit_set(tset2,numerator[temp_idx:curing_idx])
	##num3 = self.extract_fit_set(tset3,numerator[curing_idx:])
	#
	##denom1 = self.extract_fit_set(tset1,denominator[:temp_idx])
	##denom2 = self.extract_fit_set(tset2,denominator[temp_idx:curing_idx])
	##denom3 = self.extract_fit_set(tset3,denominator[curing_idx:])
	#
	#numerator, denominator = [], []
	#
	if curing_idx!=None:
	    numerator = list(num1)[:curing_idx-win1] + list(num2)[curing_idx-win1:]
	    #
	    denom1 = list(denom1_1[:curing_idx-win1]) + list(denom1_2)[curing_idx-win1:]
	    denom2 = list(denom2_1)[:curing_idx-win1] + list(denom2_2)[curing_idx-win1:]
	else:
	    numerator, denom1, denom2 = list(num1), list(denom1_1), list(denom2_1)
	
	#ofile = open('factors.csv','w')
	#
	#for idx in xrange(0,len(denom1)):
	#    print >> ofile, self.time_set[idx],',',numerator[idx],',',(denom1[idx]-denom2[idx]),',',denom1[idx],',',denom2[idx]
	#
	#ofile.close()
	
	return numerator, denom1, denom2, index_set

    def write_filtered_inputs(self,data):
	#curing_idx = self.get_curing_idx()
	#
	#win1, win2 = 20, 50
	#
	#data1 = signal.wiener(data,win1)
	#
	#data2 = signal.wiener(data,win2)
	#
	#data = list(data1)[:curing_idx+win1] + list(data2)[curing_idx+win1:]
	
	return data
		
    def filter_exponent(self,exponent,cutoff_idx):	
	win1, win2 = self.denoising_window, 5*self.denoising_window
	
	#exp1 = signal.wiener(exponent,win1)
	exp1 = signal.savgol_filter(exponent[:cutoff_idx],win1,1)
	#exp2 = signal.wiener(exponent,win2)
	exp2 = signal.savgol_filter(exponent[cutoff_idx:],win2,1)
	
	exponent = []
	
	#exponent = list(exp1)[:curing_idx+win1] + list(exp2)[curing_idx+win1:]
	exponent = list(exp1)+ list(exp2)
	
	return exponent