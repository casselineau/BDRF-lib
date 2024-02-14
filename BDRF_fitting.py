import numpy as N
import matplotlib.pyplot as plt
from sys import path
from BDRF_models import Cook_Torrance
from scipy.optimize import minimize, curve_fit
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator

from sklearn.neighbors import NearestNeighbors

from os import makedirs
from os.path import exists

import gc

'''
Regroups functions to perform a fitting of the incidence plane spectral BDRF with a microfacet model (Cook-Torrance is implemented so far) and obtain a microfacet-model based approximation of the full spectral BDRF.
'''
def build_inc_interpolator(thetas_r, phis_r, bdrf):
	'''
	This is only valid for a single incidence direction and a single wavelength or spectrally integrated bdrf.
	'''
	points = N.array([thetas_r, phis_r]).T
	points_tri = Delaunay(points)

	interpolator = LinearNDInterpolator(points_tri, bdrf)

	return interpolator

def clean_data(thetas, phis, data, K=6, deshade=True):
	'''
	a) Fill Nan values from angular distance weighted average of K nearest neighbours.
	b) Find and removes the error data due to beam shading form the arms by looking at the gradients with the KNNs.
	K must be 4 or higher.
	'''
	sin_t = N.sin(thetas)
	cos_t = N.cos(thetas)
	nans = N.isnan(data)
	# Calculate the angular distrance of the bad points to good poinst and interpolate based onm angular distance with the three closest points
	for i,d in enumerate(data):
		if nans[i] or deshade:
			psis = N.arccos(N.sin(thetas[i])*sin_t+N.cos(thetas[i])*cos_t*N.cos(phis[i]-phis))
			good_psis = psis[~nans]
			# Find non-nan K nearest points
			closest = N.argsort(good_psis)[1:K+1]
			weights = 1./good_psis[closest]
			# If d is NaN, we find the K nearest neighbours that are not NaNs and interpolate
			if nans[i]:
				# Interpo;late (weighted average):
				data[i] = N.sum(weights*data[~nans][closest])/N.sum(weights)
			elif deshade: 
				# if the d is a shaded point, it should have a value significanly lower than at least K-2 (the points are measured in a spiral and 2 KNNs could still be errors because of the arm traject0ry).
				#gradients = (data[~nans][closest]-d)*weights
				#pos_grads = gradients > 0.
				good_data = data[~nans][closest]
				below = good_data > d
				if N.sum(below)>(K-2):
					weights = weights[below]
					data[i] = N.sum(weights*good_data[below])/N.sum(weights)

	return data

def get_incidence_plane_bdrf(th_i, phi_i, thetas_r, phis_r, bdrf, th_rs, deshade=True):
	'''
	This is only valid for a single incidence direction and a single wavelength or spectrally integrated bdrf.
	th_i, phi_i - incident theta and phi angles
	thetas_r, phis_r - thetas and phi angles of the 2D bdrf data.
	bdrf - bdrf data corresponding to the thetas_r and phis_r angles
	th_rs - the incidence plane angles asked to the interpolator.
	deshade - wether or not to use the shaded data cleaning in the pre-processing algorithm that otherwise only removes and interpolates NaNs.
	'''
	interpolator = build_inc_interpolator(thetas_r, phis_r, bdrf)

	phi_r = N.ones(len(th_rs))*phi_i
	in_plane_bdrf = N.zeros(len(th_rs))

	neg = th_rs<0.
	phi_r[~neg] = (phi_i+N.pi)%(2.*N.pi) # positive thetas are the forward scattered directions, negative ones are the backward scattered direction

	in_plane_bdrf[neg] = interpolator(-th_rs[neg], phi_r[neg])
	in_plane_bdrf[neg] = clean_data(-th_rs[neg], phi_r[neg], in_plane_bdrf[neg], deshade=True)
	in_plane_bdrf[~neg] = interpolator(th_rs[~neg], phi_r[~neg])
	in_plane_bdrf[~neg] = clean_data(th_rs[~neg], phi_r[~neg], in_plane_bdrf[~neg], deshade=True)
	
	# Zero interpolation is problematic with this interpolator and coordinate system, so we correct it manually
	in_plane_bdrf[th_rs==0] = (in_plane_bdrf[neg][-1]+in_plane_bdrf[~neg][1])/2.
	return th_rs, in_plane_bdrf

def CT_wrap(inc_angle_rad, det_angle_rad, m, R, alpha):
	'''
	Calculate the incident plane CT model output for angular inputs.
	'''
	d_i = -N.array([N.sin(inc_angle_rad), 0., N.cos(inc_angle_rad)])
	d_n = N.array([0.,0.,1.])
	d_r = N.array([N.sin(det_angle_rad), 0, N.cos(det_angle_rad)])
	return Cook_Torrance(d_i, d_r, d_n, m, R, alpha)

def CT_fit(inc_angle_rad, det_angle_rad, ref_tot):
	'''
	Fit the CT model to measured data for a set of angles.
	'''
	var0 = N.array([1.1, 0.0001, 0.4]) #Initial conditions: m, R_dh_Lam, alpha
	fixed_var = [inc_angle_rad, det_angle_rad, ref_tot]
	# Optimisation bounds
	bounds = N.array([[1.0, 1.5], [0., N.amin(ref_tot)], [0., 5.]])

	# Optimisation constraints
	constraints = [{'type':'ineq', 'fun': lambda x: x-bounds[:,0]}, {'type':'ineq', 'fun': lambda x: bounds[:,1]-x}]
	# Results
	res = minimize(least_square_diff, var0, args=fixed_var, method='COBYLA', constraints=constraints, options={'tol':1e-6})
	print(res.x, res.fun, res.success)
	m, R_dh_Lam, alpha = res.x

	CT = []
	for i, d in enumerate(det_angle_rad):
		CT.append(CT_wrap(inc_angle_rad, d, m, R_dh_Lam, alpha))		
		
	return m, R_dh_Lam, alpha, N.array(CT)
	
def CT_curve_fit(inc_angle_rad, det_angle_rad, ref_tot, daq=None):
	'''
	Fit the CT model to measured data for a set of angles.
	'''
	var0 = N.array([1.03, 0.0001, 0.4]) #Initial conditions: m, R_dh_Lam, alpha
	fixed_var = [inc_angle_rad, det_angle_rad, ref_tot]
	# Optimisation bounds
	bounds = (N.array([1.0, 0.0, 0.1]), N.array([1.1, .002, 2.]))
	
	if daq is not None:
		# Uncertainty ( from Moritz: /media/ael/Flashy/backup_05-06-2021/Documents/Boulot/Projects/HEASeRS/FISE/Goniometer_Moritz/make_griddata_edit.py)
		# uncertainty on current
		#s_daq0 = 3.3 * 1e-6 # pA  -- this is the standard deviation
		s_daq0 = 8 * 1e-6 # pA  --  this adds 5pA uncert on offset
		s_daq = .02 * daq + s_daq0  # this is mainly a systematic error on the reference beam measurement
		# use correct normalization (BRDF column makes no sense for nosample measurements!!!)
		phi_0 = 0.00871  # median of 7 nosample measurements 05762 - 05848
		BRDF = daq / N.cos(det_angle_rad) / phi_0	
		# considering both s_daq and s_th_r
		deg = N.pi/180.
		s_th_r = .3*deg  # uncercainty on th_r (mainly due to spot size + sensor size)
		u = N.sqrt(daq**2 * s_th_r**2 * N.tan(det_angle_rad)**2 + s_daq**2)
		d = N.cos(det_angle_rad) * phi_0
		s_BRDF = u / d
		# Results
		popt, pcov = curve_fit(f=CT_param, xdata=det_angle_rad, ydata=ref_tot, p0=var0, sigma=s_BDRF, absolute_sigma=True, bounds=bounds)
	else:
		s_BDRF = 1/N.cos(det_angle_rad)/10.
		popt, pcov = curve_fit(f=CT_param, xdata=det_angle_rad, ydata=ref_tot, p0=var0, sigma=s_BDRF, absolute_sigma=False, bounds=bounds)
	m, R_dh_Lam, alpha = popt

	CT = []
	for i, d in enumerate(det_angle_rad):
		CT.append(CT_wrap(inc_angle_rad, d, m, R_dh_Lam, alpha))		
		
	return m, R_dh_Lam, alpha, N.array(CT)
	
def CT_param(var, fixed_var):
	m, R_dh_Lam, alpha = var
	inc_angle, det_angles, bdrfs_mes = fixed_var
	CT = []
	for i, d in enumerate(det_angles):
		CT.append(CT_wrap(inc_angle, d, m, R_dh_Lam, alpha))
	return N.array(CT)
	
def least_square_diff(var, fixed_var):
	m, R_dh_Lam, alpha = var
	inc_angle, det_angles, bdrfs_mes = fixed_var
	CT = []
	for i, d in enumerate(det_angles):
		CT.append(CT_wrap(inc_angle, d, m, R_dh_Lam, alpha))
	norm_bdrf = N.amax(bdrfs_mes)
	lsd = N.trapz((N.array(bdrfs_mes)-N.array(CT))**2)
	return lsd
