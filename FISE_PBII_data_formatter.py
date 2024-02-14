import numpy as N
from sys import path
from os import listdir
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator

rad = N.pi/180.
# Script to load the full measurement for one sample and save the data in a CSV format suitable for the rest of the work.
def convert_to_BDRF(sample_file, dark_current_sample_files, source_power):
	'''
	BDRF is here equal to the sensor measurement of the sample minus the dark current of the sensor for this measurement divided by the solid angle subtended by the sensor and by the integrated sensor measurement of the lamp multiplied by the cosin of the detector theta angle.
	As measurements are not always at the exact same position, interpolations are necessary using interpolate_to().
	'''
	# Load data:
	# Measurement:
	bdrf = N.loadtxt(sample_file, delimiter=',', skiprows=1)
	# Dark current data:
	dark_data = []
	for i, dark_file in enumerate(dark_current_sample_files):
		dark_data.append(N.loadtxt(dark_file, delimiter=',', skiprows=1))
	# Integrate averaged lamp data. First we need to interpolate to ba able to use composition trapezoidal rule. The interpolation is directly done on the desired grid:
	for i, dark in enumerate(dark_data):
		i_dcl = interpolate_to(bdrf, dark)
		bdrf[:,2] -= N.copy(i_dcl[:,2])/float(len(dark_data))
	bdrf[:,2] = bdrf[:,2]/(N.cos(bdrf[:,0]*rad)*source_power)
	N.savetxt(sample_file[:-4]+'_BDRF.csv', bdrf, header='theta_r (deg), phi_r (deg), BDRF', delimiter=',')
	return bdrf

def process_lamp(lamp_file, dark_current_lamp_files, dang=None):
	'''
	1 - load the lamp and dark curent data
	2 - interpolate the dark currents to the position of the lamp measurements and subtract the average of all files.
	3 - integrate the lamp data.
	'''
	# LOAD:
	# Lamp:
	lamp_data = N.loadtxt(lamp_file, delimiter=',', skiprows=1)
	# Dark current lamp data:
	dark_lamp_data = []
	for i, dark_lamp_file in enumerate(dark_current_lamp_files):
		dark_lamp_data.append(N.loadtxt(dark_lamp_file, delimiter=',', skiprows=1))
	# Interpolate the dark currents to grid of the lamp data subtract the contribution to the average:
	for i, dark_lamp in enumerate(dark_lamp_data):
		i_dc = interpolate_to(lamp_data, dark_lamp)
		lamp_data[:,2] -= i_dc[:,2]/float(len(dark_lamp_data))

	# Integrate the corrected lamp signal:
	if dang is not None:
		source_power = integrate_polar(lamp_data, dang)
	else:
		source_power = integrate_polar(lamp_data)
	return source_power_signal
	

def interpolate_to(data_ref, data_to_int):
	'''
	Take data from data_to_int, build a linear interpolator and interpolate to the points given in data_ref.
	data_ref - the reference data, the grid of thsi data is used as targetted points of interpolation.
	data_to_int - The data to interpolate.
	
	Data format is the fromat produced by format_data().
	'''
	th_ref, phi_ref = data_ref[:,:2].T
	
	# Find the points in the data_ref that are outside of the data_to_ref domain of definition and will lead to NaNs.
	th_min, th_max = N.amin(data_to_int[:,0]), N.amax(data_to_int[:,0])
	phi_min, phi_max = N.amin(data_to_int[:,1]), N.amax(data_to_int[:,1])
	bad_th = N.logical_or(th_ref<th_min, th_ref>th_max)
	bad_phi = N.logical_or(phi_ref<phi_min, phi_ref>phi_max)
	bad_points = N.logical_or(bad_th, bad_phi)
	# Add these points to data_to_ref and interpolate with nearest 3 neighbours to have a value there.
	new_to_int = data_ref[bad_points]
	new_to_int[:,2] = 0
	# Calculate the angular distrance of the bad points to good poinst and interpolate based onm angular distance with the three closest points
	sint = N.sin(data_to_int[:,0])
	cost = N.cos(data_to_int[:,0])
	for i, p in enumerate(new_to_int):
		psis = N.arccos(N.sin(p[0])*sint+N.cos(p[0])*cost*N.cos(p[1]-data_to_int[:,1]))
		# Find nearest 3 points
		closest = N.argsort(psis)[:3]
		# Interpo;late (weighted average):
		weights = 1./psis[closest]
		new_to_int[i,2] = N.sum(weights*data_to_int[closest,2])/N.sum(weights)
	# Add new points:
	data_to_int = N.concatenate((data_to_int, new_to_int))

	# don't forget the sin(th) to triangulate properly:
	points = N.array([data_to_int[:,0], N.sin(data_to_int[:,0])*data_to_int[:,1]]).T
	points_tri = Delaunay(points)
	interpolator = LinearNDInterpolator(points_tri, data_to_int[:,2])

	interpolated_data = interpolator(th_ref, phi_ref*N.sin(th_ref))

	return N.array([th_ref, phi_ref, interpolated_data]).T

def integrate_polar(spatial_data, dang=N.pi/1000.):
	'''
	Fully integrate the 3D data over the definition domain
	'''
	# convert spatial data to radianst
	spatial_data[:,:2] *= rad
	# Interpolate to a regular grid in polar:
	minth, maxth = N.amin(spatial_data[:,0]), N.amax(spatial_data[:,0])
	thetas_i, phis_i = N.linspace(minth, maxth, N.ceil((maxth-minth)/dang)), N.linspace(-N.pi, N.pi, 2.*N.pi/dang)
	TH, PH = N.meshgrid(thetas_i, phis_i)
	th, ph = N.hstack(TH), N.hstack(PH)
	spatial_data_i = interpolate_to(N.array([th, ph, N.zeros(len(th))]).T, spatial_data)

	data_P = N.zeros(len(phis_i))
	# First integrate at constant phis (along thetas):
	for i, p in enumerate(phis_i):
		select = spatial_data_i[:,1] == p
		ths = spatial_data_i[select,0]
		data_P[i] = N.abs(N.trapz(spatial_data_i[select,2]*N.sin(ths), ths))
	# Then integrate the results along phi:
	data_T = N.trapz(data_P, phis_i)
	return data_T

def identify_all_data(loc):
	'''
	Read through a directory to find and label the measurement files.
	'''
	stuff = listdir(loc)
	sample_files, dark_current_lamp_files, dark_current_sample_files = [], [], []
	for s in stuff:
		if '.txt' in s:
			if 'Lamp_reference' in s:
				lamp_file = loc+'/'+s
			if 'DV' in s:
				sample_files.append(loc+'/'+s)
			if 'Dark_current_Lamp' in s:
				dark_current_lamp_files.append(loc+'/'+s)
			if 'Dark_current_hemisphere' in s:
				dark_current_sample_files.append(loc+'/'+s)
	return lamp_file, sample_files, dark_current_lamp_files, dark_current_sample_files

def format_data(sample_file, target=None):
	'''
	Load PBII output data and save in a CSV file.
	'''
	with open(sample_file, 'r') as fo:
		data  = file.readlines(fo)

	header = True
	while header:
		if data[0][0] == '#':
			data = data[1:]
		else:
			header = False

	for l in range(len(data)):
		data[l] = [float(d) for d in data[l].split('\r')[0].split('\t')]
	data = N.array(data)

	if target is None:
		target = sample_file[:-3]+'csv'

	N.savetxt(target, data, delimiter=',', header='theta (deg), phi (deg), sensor (uA)')
	return target

if __name__ == "__main__":
	"""
	Arguments:
	sample_file target

	When new data is added:
	1 - Convert all the data into csv with the right content.
	2 - Process the lamp data to get the integrated total signal.
	2 - Calculate BDRF based on the full mesurement data including dark currents etc
 	"""
	from sys import argv

	loc = argv[1]
	#if len(argv)>3:
	#	target = argv[3]
	#else:
	#	target = None

	lamp_file, sample_files, dark_current_lamp_files, dark_current_sample_files = identify_all_data(loc)
	'''
	lamp_file = format_data(lamp_file)


	dangs = N.pi/N.array([200, 300, 400, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000])
	corrected_lamp_data = []
	lamp_file = format_data(lamp_file)
	for i,f in enumerate(dark_current_lamp_files):
		dark_current_lamp_files[i] = format_data(f)
	for dang in dangs:
		print i
		corrected_lamp_data.append(process_lamp(lamp_file, dark_current_lamp_files, dang))
	print corrected_lamp_data
	N.savetxt(path[0]+'/lamp_integration_test.csv', N.array([dangs, corrected_lamp_data]).T, delimiter=',', header='delta_ang (rad), integrated lamp signal (ua)')
	
	import matplotlib.pyplot as plt
	dangs, corrected_lamp_data = N.loadtxt(path[0]+'/lamp_integration_test.csv', delimiter=',', skiprows=1).T
	plt.figure()
	plt.plot(dangs, corrected_lamp_data)
	plt.xlabel(r'${\Delta\theta}$ (rad)')
	plt.ylabel(r'Integrated lamp signal (uA)')
	plt.savefig(path[0]+'/source_integration.png')
	'''
	source_power = N.loadtxt(path[0]+'/lamp_integration_test.csv', delimiter=',', skiprows=1).T[1,-1]
	for i,f in enumerate(dark_current_sample_files):
		dark_current_sample_files[i] = format_data(f)
	for sample_file in sample_files:
		sample_file = format_data(sample_file)
		print 'processing', sample_file
		convert_to_BDRF(sample_file, dark_current_sample_files, source_power)

		
	#'''	
