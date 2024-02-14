import numpy as N
from spectral_processing import wavelength_to_rgb
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator

rad = N.pi/180.

def plot_inplane_BDRF(wavelength, thetas_r_deg, theta_i_deg, bdrfs, diffuse_ratio=False, normalised=False, **kwargs):
	'''
	Plots the BDRF in the plane of incidence.
	This function only makes the plot in an active pyplot figure, the figure creation and saving is done extrenally, in the script dedicated to data processing.
	Arguments:
	- wavelength: in nm (this argument is only used here to color the lines with a realistic color using the wavelength_to_rb_function) two options
		- A single value for the exact wavelength of interest
		- a list of two values determining the range of the data
	- thetas_r_deg: reflected polar angles in degrees. Negative values are understood.
	- theta_i: angle of incidence of the beam in degrees. /!\ Important: in the plot, the incident angle is plotted in negative values of theta.
	- bdrfs: bi-directional reflectance measurements corresponding to this incident angle and reflected angles.
	- label: label of the bdrf curve.
	- diffuse_ratio: an optional argument. If set to True, integrates teh arcual BRDF evaluates how "diffuse" the in-plance bdrf is compared to the equivalent diffuse reflectance.
	- normalised: optional argument, if set to True, the bdrf is normalised to the maximum value measured.
	'''
	theta_i_deg = -theta_i_deg # we choose to ploit the incident angle in the negative values.
	thetas, theta_i = thetas_r_deg*rad, theta_i_deg*rad

	if diffuse_ratio:
		dar = N.trapz(bdrfs*N.cos(thetas), thetas) # directional-arcuate reflectance
		brdf_diff = dar/2. # integration over [-pi2, pi/2] of cos(t)dt, the arc, leads to this
		bdrfs /= brdf_diff # ratio: over 1 is higher than equivalent diffuse, under 1 is lower.
	if normalised:
		bdrfs /= N.amax(bdrfs)

	bdrf_int = interp1d(thetas, bdrfs) # makes the plot look better by interpolating lineraly between angles. Otherwise, the plot will make a straight line in polar coordinates, which is wrong.
	thetas_plot = N.linspace(N.amin(thetas), N.amax(thetas), 10*len(thetas)+1)
	if 'color' not in kwargs.keys():
		if wavelength is not None:
			color = wavelength_to_rgb(wavelength)
			kwargs.update({'color':color})
	else:
		color = kwargs['color']

	plt.plot(thetas_plot, bdrf_int(thetas_plot), zorder=100, **kwargs)
	#plt.yscale('log')
	plt.xlim(-90, 90.)
	plt.xlabel(r'${\theta_\mathrm{r}}$ ($^\circ$)')
	plt.ylabel(r'${\rho^{\prime\prime}}$')
	plt.gca().set_theta_offset(N.pi/2.)
	plt.gca().set_theta_direction(-1)
	return bdrf_int

def plot_BDRF(wavelength, thetas_r_deg, phis_r_deg, bdrfs, th_i_deg, phi_i_deg, saveloc, thmax_deg=None, specularity_indicators=True):

	th_rs, phi_rs = thetas_r_deg*rad, phis_r_deg*rad

	plt.figure()
	plt.subplot(111, projection='polar')
	plt.suptitle('${\lambda}$='+str(wavelength))

	points = N.array([th_rs, N.sin(th_rs)*phi_rs]).T
	points_tri = Delaunay(points)
	interpolator = LinearNDInterpolator(points_tri, bdrfs)

	if thmax_deg is None:
		thmax_deg = 90.
		
	plt.ylim(ymax=thmax_deg)		
	
	ths, phs = N.linspace(0., thmax_deg, 200)*rad, N.linspace(0., 360., 400)*rad
	TH, PHI = N.meshgrid(ths, phs)
	# Original interpolator
	BDRF = interpolator(TH, PHI*N.sin(TH))

	# Fill Nan values that appear where the interpolated data is outside of the domain of definition of the measurement (in angular coordiantes). This is performed using inverse of distance weighted average of KNN (NN optional argument can be changed).
	bdrfs = clean_data(TH, PHI, BDRF)

	plt.pcolormesh(PHI, TH/rad, BDRF)#, vmin=0.)#, vmax= 0.1)
	plt.colorbar(label=r'${\rho^{\prime\prime}}$')

	NANS = N.isnan(BDRF)
	plt.scatter(PHI[NANS], TH[NANS]/rad, marker='*', color='darkred')

	if specularity_indicators:
		plt.scatter(phi_i_deg*rad, th_i_deg, marker='o', color='r')
		plt.scatter(N.pi+phi_i_deg*rad, th_i_deg, marker='+', color='r', s=50)
	plt.savefig(saveloc)
	plt.close()
