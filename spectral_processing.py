import numpy as N
from sys import path

def wavelength_to_rgb(wavelength, gamma=0.8):
	''' 
	taken from http://www.noah.org/wiki/Wavelength_to_RGB_in_Python
	This converts a given wavelength of light to an 
	approximate RGB color value, for graphical purposes. The wavelength must be given
	in nanometers in the range from 380 nm through 750 nm
	(789 THz through 400 THz).

	Based on code by Dan Bruton
	http://www.physics.sfasu.edu/astro/color/spectra.html
	Additionally alpha value set to 0.5 outside range

	TODO: Use this instead: https://www.fourmilab.ch/documents/specrend/: the real processing based on CIE standard.
	'''
	if len(N.hstack([wavelength]))>1:
		wavelength = N.average(wavelength)
	wavelength = float(wavelength)
	if wavelength >= 380 and wavelength <= 750:
		A = 1.
	if wavelength < 380:
		A = wavelength/380.
		R = 0.3**gamma
		G = 0.
		B = 0.3**gamma
	elif wavelength >= 380 and wavelength <= 440:
		attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
		R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
		G = 0.0
		B = (1.0 * attenuation) ** gamma
	elif wavelength >= 440 and wavelength <= 490:
		R = 0.0
		G = ((wavelength - 440) / (490 - 440)) ** gamma
		B = 1.0
	elif wavelength >= 490 and wavelength <= 510:
		R = 0.0
		G = 1.0
		B = (-(wavelength - 510) / (510 - 490)) ** gamma
	elif wavelength >= 510 and wavelength <= 580:
		R = ((wavelength - 510) / (580 - 510)) ** gamma
		G = 1.0
		B = 0.0
	elif wavelength >= 580 and wavelength <= 645:
		R = 1.0
		G = (-(wavelength - 645) / (645 - 580)) ** gamma
		B = 0.0
	elif wavelength >= 645 and wavelength <= 750:
		attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
		R = (1.0 * attenuation) ** gamma
		G = 0.0
		B = 0.0
	elif wavelength > 750:
		A = 750./wavelength
		R = 0.3**gamma
		G = 0.
		B = 0.

	return (R,G,B,A)


def spectral_weights(wl, prop, spectrum):
	'''
	Spectral weighting at wl locations property.
	First the weights_spectrum distribution is established based on the reference spectrum, then the weighst are interpolated at the given wl locations.
	We do this this way because it makes it possible to take into account spectral variations in the reference spectrum that would otherwise be ignored if we only interpolated at the data spectrum wavelengths.
	'''
	data = N.loadtxt(spectrum)
	wl_data = data[:,0]
	q_data = data[:,1]

	weights_spectrum = q_data/N.trapz(wl_data, q_data)
	# Interpolation to correclty evaluate the spectral weights if the data and the reference spectrum and the wl are not aligned.
	weights_wl = N.interp(wl, wl_data, weights_spectrum, left=0., right=0.)
	return weights_wl

def solar_weighted_absorptance(wl, abso, solar_spectrum='/media/ael/Flashy/backup_05-06-2021/Documents/Boulot/Material_properties/astmg173.txt'):
	'''
	Solar weighted absorptance. 
	Arguments:
	- wl: wavelengths in nm
	- abso: spectral absorptance in decimal format
	- solar_spectrum: .txt file of two columns. First comulmn is wavelengths and second one the spectral radiance. Standard 1.5 air mass spectrum is found at: https://www.nrel.gov/grid/solar-resource/spectra-am1.5.html
	Returns:
	- solar-weighted absorptance array and weights
	'''
	weights = spectral_weights(wl, abso, solar_spectrum)
	return abso*weights, weights

def solar_weighted_reflectance(wl, ref, solar_spectrum='/media/ael/Flashy/backup_05-06-2021/Documents/Boulot/Material_properties/astmg173.txt'):
	'''
	Solar weighted absorptance. 
	Arguments:
	- wl: wavelengths in nm
	- ref: spectral reflectance in decimal format
	- solar_spectrum: .txt file of two columns. First comulmn is wavelengths and second one the spectral radiance. Standard 1.5 air mass spectrum is found at: https://www.nrel.gov/grid/solar-resource/spectra-am1.5.html
	Returns:
	- solar-weighted reflectance array and weights
	'''
	weights = spectral_weights(wl, ref, solar_spectrum)
	return ref*weights, weights

def planck_weights(wl, T):
	'''
	Planck (blackbody) -weighted spectral radiative property. Directly use the Planck law to spectrally weight, no need to do any interpolation.
	Arguments:
	- wl: wavelengths in nm
	- T: temperature in K
	Returns:
	- Planck-weighted property
	'''
	h = 6.626070040e-34 # Planck constant
	c = 299792458. # Speed of light in vacuum
	k = 1.38064852e-23 # Boltzmann constant
	norm = 5.67e-8*(T**4.) # Stefan-Boltzmann law
	weights = (2.*N.pi*h*c**2.)/((wl)**5.)/(N.exp(h*c/(wl*k*T))-1.)/norm
	return weights

def planck_weighted_absorptance(wl, abso, T):
	'''
	Planck (blackbody) -weighted absorptance. Directly use the Planck law to spectrally weight, no need to interpolations.
	Arguments:
	- wl: wavelengths in nm
	- abso: spectral absorptance in decimal format
	- T: temperature in K
	Returns:
	- Planck-weighted absortance
	'''
	weights = planck_weights(wl, T)
	return abso*weights, weights

def co_interpolate_spectra(wl1, data1, wl2, data2):
	'''
	Utility function to alight two spetra on the same regular wavelengths scale, defined based on teh number of points in the spectrum with more spectral definition.
	Arguments:
	wl1, data1: wavelengths and data of the first spectrum
	wl2, data2: wavelengths and data of the second spectrum
	Returns:
	- wl_inter: interpolation wavelengths
	- data1_inter, data2_inter: interpolated spectra corresponding to wl_inter 
	'''
	# sort stuff (just in case...)
	wl1, data1 = N.sort(wl1), data1[N.argsort(wl1)]
	wl2, data2 = N.sort(wl2), data2[N.argsort(wl2)]
	# define interpolation range:
	xmin, xmax = N.amin(N.hstack([wl1, wl2])), N.amax(N.hstack([wl1, wl2]))
	wl_inter = N.linspace(xmin, xmax, N.amax([len(wl1), len(wl2)]))
	# Interpolate spectra
	data1_inter = N.interp(wl_inter, wl1, data1)
	data2_inter = N.interp(wl_inter, wl2, data2)

	return wl_inter, data1_inter, data2_inter

def diff_spectra(wl1, data1, wl2, data2):
	wl_inter, data1_inter, data2_inter = co_interpolate_spectra(wl1, data1, wl2, data2)
	'''
	Utility function: co-interpolates and makes the difference between two spectra.
	Arguments:
	wl1, data1: wavelengths and data of the first spectrum
	wl2, data2: wavelengths and data of the second spectrum
	Returns:
	- wl_inter: interpolation wavelengths
	- ref1_inter - ref2_inter: difference of the interpolated spectra.
	'''
	return wl_inter, ref1_inter - ref2_inter

def process_spectrum(wl, data, clean=None, smooth=None):
	'''
	Filtering function to clean noise.
	Arguments:
	- wl: wavelengths in nm
	- data: spectral data
	- clean: list of lists with starting and ending wl of spiked areas
	- smooth: list of smoothing instructions: [[lambda0, lambda1, kernel size]] 
		- lambda0, lambda1: stant and end of the smoothing region
		- kernel size: number of points for the moving average
	'''
	if clean is not None:
		wl_r = N.copy(wl)
		data_r = N.copy(data)
		for c in clean:
			remove = N.logical_and(wl_r>=c[0], wl_r<=c[1])
			wl_r = wl_r[~remove]
			data_r = data_r[~remove]
		data = N.interp(wl, wl_r, data_r)

	if smooth is not None:
		for s in smooth:
			kernel = N.ones(s[2])/s[2]
			interval = N.logical_and(wl>=s[0], wl<=s[1])
			datai = N.convolve(data[interval], kernel, mode='same')
			# Edge effects of the convolution
			datai[:len(kernel)/2] = datai[len(kernel)/2]
			datai[-len(kernel)/2:] = datai[-len(kernel)/2]
			data[interval] = datai
	return data
		
	

