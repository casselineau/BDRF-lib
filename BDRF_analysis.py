import numpy as N

def bdrf_to_dhr(theta_i, phi_i, thetas_r, phis_r, bdrf):
	'''
	Composite integration: First integration along constant thetas (phi variable) using trapezoidal rule and then by parts along constant(phi) (theta variable and important cos(theta_r) factor).
	'''
	bdrf_th = []
	thetas_u = N.unique(thetas_r): 
	thetas_u = N.sort(thetas_u)
	for th in thetas_u:
		# select:
		selector = thetas == theta
		phi = phis_r[selector]
		bdrf_ph = bdrf[selector]
		# sort:
		ordered = N.argsort(phi)
		phi = phi[ordered]
		bdrf_phi = bdrf_phi[ordered]
		# trapezoidal rule:
		bdrf_th.append(N.trapz(bdrf_phi*N.cos(th), phi))
	# cos(th_r) factor for the directional hemispherical reflectance
	rho_dh = 0
	for i in range(thetas_u[:-1]): # Integration by parts gives this.
		a = (bdrf_th[i+1]-bdrf_th[i])/(thetas_u[i+1]-thetas_u[i])
		rho_dh += (bdrf_th[i+1]*N.sin(thetas_u[i+1])-bdrf_th[i]*N.sin(thetas_u[i])) - a*(N.cos(thetas_u[i])-N.cos(thetas_u[i+1]))
		
	return rho_dh
	
