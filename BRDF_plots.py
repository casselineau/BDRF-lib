import numpy as N
from spectral_processing import wavelength_to_rgb
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def plot_inplane_BRDF(wavelength, thetas_r_deg, inc_angle_deg, bdrfs,  diffuse_ratio=False, normalised=False, **kwargs):
	'''
	Plots the BDRF in the plane of incidence.
	This function only makes the plot in an active pyplot figure, the figure creation and saving is done extrenally, in the script dedicated to data processing.
	Arguments:
	- wavelength: in nm (this argument is only used here to color the lines with a realistic color using the wavelength_to_rb_function) two options
		- A single value for the exact wavelength of interest
		- a list of two values determining the range of the data
	- thetas_r_deg: reflected polar angles in degrees. Negative values are understood.
	- inc_angle_deg: angle of incidence of the beam in degrees.
	- bdrfs: bi-directional reflectance measurements corresponding to this incident angle and reflected angles.
	- label: label of the bdrf curve.
	- diffuse_ratio: an optional argument. If set to True, integrates teh arcual BRDF evaluates how "diffuse" the in-plance bdrf is compared to the equivalent diffuse reflectance.
	- normalised: optional argument, if set to True, the bdrf is normalised to the maximum value measured.
	'''
	thetas, inc_angle = thetas_r_deg/180.*N.pi, inc_angle_deg/180.*N.pi

	if diffuse_ratio:
		dar = N.trapz(bdrfs*N.cos(thetas), thetas) # directional-arcuate reflectance
		brdf_diff = dar/2. # integration over [-pi2, pi/2] of cos(t)dt, the arc, leads to this
		bdrfs /= brdf_diff # ratio: over 1 is higher than equivalent diffuse, under 1 is lower.
	if normalised:
		bdrfs /= N.amax(bdrfs)

	bdrf_int = interp1d(thetas, bdrfs) # makes the plot look better by interpolating lineraly between angles. Otherwise, the plot will make a straight line in polar coordinates, which is wrong.
	thetas_plot = N.linspace(N.amin(thetas), N.amax(thetas), 10*len(thetas)+1)

	plt.plot(thetas_plot, bdrf_int(thetas_plot), color=wavelength_to_rgb(wavelength), zorder=100, **kwargs)
	plt.gca().set_theta_offset(N.pi/2.)
	plt.gca().set_theta_direction(-1)

	plt.xticks(N.linspace(-N.pi/2., N.pi/2., 19))

