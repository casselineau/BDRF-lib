import numpy as N
import matplotlib.pyplot as plt
from sys import path
from BRDF_plots import plot_inplane_BRDF

from os import makedirs
from os.path import exists

'''
This file contains the functiosn necessary to plot the CARY UMA data from UPV.
First there are a few "switces" that, if give 1, or "True" value, will launch the scripts found after the functions definitions when running this file in Python.
Then, the functions necessary to load and plot the data are defined. The functions use a few dependencies imported at the top of the file.
Finally, examples of plotting scripts are given after teh functions definitions.
'''
# General plotting instruction to reduce fontsizes
plt.rc({'size':8})

# Switches:
plot_mes = 0
plots_sim = 0

# Functions:
def load_CARY_data(csv_file):
	'''
	Load data directly form the .csv output file of the CARY UMA and create an output list:
	Argument:
	- csv_file: csv_file output form the CARY UMA.
	Returns:
	- list of data arrays. Each element in each array corresponds the the same measurement:
		- det_angle: oriented polar angle of the detector in degrees, in the plane of incidence.
		- polarization: polarization of the incident beam in this data point.
		- wavelength_nm: wavelength of the monochromator for this data point, in nanometers
		- raw_signal/baseline_signal: bdrf measurement.
	'''
	data = N.loadtxt(csv_file, delimiter=',', skiprows=1)
	det_angle, polarization, wavelength_nm, raw_signal, baseline_signal = data.T
	return [det_angle, polarization, wavelength_nm, raw_signal/baseline_signal]

def load_sim_CARY_data(csv_file):
	'''
	Loader for simulation results of 3D printed sample
	'''
	data = N.loadtxt(csv_file, delimiter=',', skiprows=1)
	det_angle, wl0, wl1, bdrf, CIs = data.T
	wavelengths = N.vstack([wl0, wl1]).T
	return [det_angle, wavelengths, bdrf]

def get_bdrf(data_cary, incident_angle_deg, wl=[0,N.inf], theta_d=[-180., 180.], integrate_wl=False):
	'''
	Processes the data extracted from the csv file to identify incomplete measurement series, isolate a range of wavelengths, detector angles and integrate in the wavelengths range if needed.
	Arguments:
	- data_cary: output of the load_CARY_data() function with all the data form the UMA measurement.
	- incident_angle_deg
	- wl: wavelength range. Defaults to full spectrum [0,N.inf]. If a single value is given, only extracts that value.
	- theta_d: detector angle, defaults to full bi-arcual space [-180., 180.]. If a single value is given, only extracts that value.
	- integrate_wl: If set to true, the data between teh two boundaries of wl is integrated. Otherwise the data of all the wavelengths in the interval is given
	Returns:
	- The BRDF data, filtered and integrated as needed.
	'''
	# Detectro angle selection:
	if len(N.hstack([theta_d])) == 1:
		theta_d = [theta_d, theta_d]
	selector_th_d = N.logical_and(data_cary[0]>=theta_d[0], data_cary[0]<=theta_d[1])

	# Polarization selction:
	selector_pol_p = data_cary[1] == 0
	if not selector_pol_p.any():
		print 'P polarization data missing'
		return None, None, None, None, None
	selector_pol_s = data_cary[1] == 90
	if not selector_pol_s.any():
		print 'S polarization data missing'
		return None, None, None, None, None

	# Wavelengths selection:
	if len(N.hstack([wl])) == 1:
		wl = [wl, wl]
	if (wl[0]<data_cary[2]).all() or (wl[1]>data_cary[2]).all():
		print 'wavelength data missing in ', wl, 'interval'
		return None, None, None, None, None
	selector_wl = N.logical_and(data_cary[2]>=wl[0], data_cary[2]<=wl[1])
	
	# Mix selectors:
	selector = selector_th_d*selector_wl
	selector_p = selector*selector_pol_p
	selector_s = selector*selector_pol_s

	# Apply selectors and correct angles:
	det_angle_p = data_cary[0][selector_p]+incident_angle_deg
	det_angle_s = data_cary[0][selector_s]+incident_angle_deg

	# Cosine factor division to obtain BDRF: 
	ref_p_pol = data_cary[3][selector_p]/N.cos(det_angle_p/180.*N.pi)
	ref_s_pol = data_cary[3][selector_s]/N.cos(det_angle_s/180.*N.pi)

	# Output in a try statement so that incomplete data does not stop teh full extraction process if put in a loop.
	try:
		ref_tot = (ref_p_pol+ref_s_pol)/2.
		det_angle = det_angle_p
		wavelength_nm = data_cary[2][selector_p]
		sort = N.argsort(det_angle)
		data = [det_angle[sort], wavelength_nm[sort], ref_tot[sort], ref_p_pol[sort], ref_s_pol[sort]]
		if integrate_wl:
			data = integrate_bdrf_wl(data)
		return data
	except:
		print 'data missing'
		return None, None, None, None, None

def integrate_bdrf_wl(data):
	'''
	Linear spectral integration of BDRF data.
	Argument:
	- data: BDRF data as it comes out of the get_bdrf() function
	Returns:
	- data: same as input but each detector angle has its spectral data integrated in thE full range given (as a consequence there is only one curve of spectral data per detector angle).
	'''
	integ_ref_tot = []
	integ_ref_p_pol = []
	integ_ref_s_pol = []
	det_angles = N.unique(data[0])
	wl_bounds = []
	for d in det_angles:
		selector = data[0] == d
		wavelengths = data[1][selector]
		ref_tot = data[2][selector]
		ref_p_pol = data[3][selector]
		ref_s_pol = data[4][selector]

		sort = N.argsort(wavelengths)
		wavelengths = wavelengths[sort]
		ref_tot = ref_tot[sort]
		ref_p_pol = ref_p_pol[sort]
		ref_s_pol = ref_s_pol[sort]

		integ_ref_tot.append(N.trapz(ref_tot, wavelengths))
		integ_ref_p_pol.append(N.trapz(ref_p_pol, wavelengths))
		integ_ref_s_pol.append(N.trapz(ref_s_pol, wavelengths))

		wl_bounds.append([N.amin(wavelengths), N.amax(wavelengths)])
	return [det_angles, wl_bounds, integ_ref_tot, integ_ref_p_pol, integ_ref_s_pol]

def plot_bdrf_mes(sample, inc_angle_deg, wavelengths, datadir, savedir, diffuse_ratio=False, normalised=False):
	'''
	UPV CARY UMA measurement plotter.
	Arguments:
	- sample: Sample file name.
	- inc_angle_deg: incident angle of the beam in degrees
	- wavelengths; list of wavelength intervals
	- datadir: sample file folder location
	- savedir: output figure folder location (name of file is automatically generated)
	- diffuse_ratio: optional argument. If true, passes it on the plot_inplane_BRDF() function in BDRF_plots.py.
	- normalised: optional argument. If true, passes it on the plot_inplane_BRDF() function BDRF_plots.py.
	Returns:
	- Nothing but plots the data and seves it at savedir location.
	'''
	plt.figure(figsize=(7, 3))
	plt.suptitle(sample+', '+r'${\theta_\mathrm{in}}$ = %s${^{\circ}}$'%(inc_angle_deg))
	plt.subplots_adjust(top=1, left=0.45, bottom=0.0, right=1)
	plt.subplot(111, projection='polar')
	csv_file = datadir+'/'+sample+'_%i.csv'%(-inc_angle_deg)
	saveloc = savedir+'/%s_%s.png'%(sample, inc_angle_deg)
	data_cary = load_CARY_data(csv_file)
	max_y = 0
	for wavelength in wavelengths:
		det_angle, wavelength_nm, ref_tot, ref_p_pol, ref_s_pol = get_bdrf(data_cary, inc_angle_deg, wavelength, integrate_wl=True)
		
		if det_angle is not None:
			plot_inplane_BRDF(wavelength, det_angle, inc_angle_deg, ref_tot, label=wavelength, diffuse_ratio=diffuse_ratio, normalised=normalised)
			max_y = N.amax(N.hstack([ref_tot, max_y]))
	plt.scatter(inc_angle_deg/180.*N.pi, 0.9*max_y, color='r')
	plt.xlim(-N.pi/2., N.pi/2.)
	plt.gca().tick_params(axis='y', labelrotation=45)
	plt.ylim(0., max_y)
	plt.text(-N.pi/4., max_y*1.3, r'$\theta_\mathrm{r}$')
	plt.text(-125*N.pi/180., max_y*0.5, r'$\rho^{\prime\prime}$')
	plt.figlegend(loc=6, ncol=2, title='$\lambda$ (nm)')

	plt.savefig(saveloc)
	plt.close()

def plot_bdrf_sim(sample, inc_angle_deg, datadir, savedir):
	'''
	Plotter for simulated BDRF of 3D printed sample.
	'''
	plt.figure(figsize=(7, 3))
	plt.suptitle(sample+', '+r'${\theta_\mathrm{in}}$ = %s${^{\circ}}$'%(inc_angle_deg))
	plt.subplots_adjust(top=1, left=0.45, bottom=0.0, right=1)
	plt.subplot(111, projection='polar')
	csv_file = datadir+'/'+sample+'_%i_sim.csv'%(inc_angle_deg)
	saveloc = savedir+'/%s_%s_sim.png'%(sample, inc_angle_deg)
	det_angle, wavelengths, bdrf = load_sim_CARY_data(csv_file)
	max_y = 0
	for i, wl in enumerate(N.unique(wavelengths[:,0])):
	
		in_wl = wl == wavelengths[:,0]
		det_angle_wl = det_angle[in_wl]
		bdrf_wl = bdrf[in_wl]
		plot_inplane_BRDF(wavelengths[i], det_angle_wl, inc_angle_deg, bdrf_wl, label=wavelengths[i], diffuse_ratio=False)
		max_y = N.amax(N.hstack([bdrf, max_y]))

	plt.scatter(inc_angle_deg/180.*N.pi, 0.9*max_y, color='r')
	plt.xlim(-N.pi/2., N.pi/2.)
	plt.gca().tick_params(axis='y', labelrotation=45)
	plt.ylim(0., max_y)
	plt.text(-N.pi/4., max_y*1.3, r'$\theta_\mathrm{r}$')
	plt.text(-125*N.pi/180., max_y*0.5, r'$\rho^{\prime\prime}$')
	plt.figlegend(loc=6, ncol=2, title='$\lambda$ (nm)')

	plt.savefig(saveloc)
	plt.close()

# Plotting scripts examples:
if plot_mes:

	savedir = path[0]+'/UPV_measurements'
	if not exists(savedir):
		makedirs(savedir)
	datadir = path[0]+'/CARY_UMA'

	wl_int = N.linspace(400, 2400, (2400-400)/100+1)
	wavelengths = N.vstack([wl_int[:-1], wl_int[1:]]).T

	samples = ['M4', 'M4-H']
	inc_angles = N.linspace(-80, 80, 17)

	for sample in samples:
		for inc_angle in inc_angles:
			print sample, inc_angle
			# incident angle in file names are actually the normal angle to the source and are in the other direction.
			plot_bdrf_mes(sample, inc_angle, wavelengths, datadir, savedir)

if plots_sim:
	from sys import path
	from os import makedirs
	from os.path import exists

	savedir = path[0]+'/test_sim_UPV'
	if not exists(savedir):
		makedirs(savedir)
	datadir = savedir
	samples = ['Hybrid_2V', 'Hybrid_2H']
	inc_angles = [-80, 0]#N.linspace(-80, 80, 17)
	for sample in samples:
		for inc_angle in inc_angles:
			print sample, inc_angle
			# incident angle in file names are actually the normal angle to the source and are in the other direction.
			plot_bdrf_sim(sample, inc_angle, datadir, savedir)

	samples = ['Hybrid_2Vbb', 'Hybrid_2Hbb']
	inc_angles = N.linspace(-80, 80, 17)
	for sample in samples:
		for inc_angle in inc_angles:
			print sample, inc_angle
			# incident angle in file names are actually the normal angle to the source and are in the other direction.
			plot_bdrf_sim(sample, inc_angle, datadir, savedir)
