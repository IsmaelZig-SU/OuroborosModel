import sys
import os
import torch
import numpy as np 
from scipy import interpolate, linalg
from tqdm import tqdm
from Data_Assimilation import Data_Assimilation
from src_param.Eval_MZA import Eval_MZA
from scipy.stats import kstest, norm
from scipy.special import erf
from scipy import interpolate, linalg
import matplotlib.pyplot as plt

class Initialisation : 

	def __init__(self, X_true, nobs) :

		"""Initialise the loop, generates first forecast ensemble and finds sensor locations

		Parameters : 

			model (object MZA_Experiement) : Stochastic model 
			X_true (np.ndarray) : True state [T, dim]
			nobs (int) : Number of observations 

		Returns : 

			H (np.ndarray) : Downsampling matrix (sensor locations) [nobs, dim]
			Y_obs (np.ndarray) : High fidelity observations (X_true@H.T) [T, nobs]

		""" 

		self.model = model 
		self.X_true = X_true
		self.nobs = nobs
		self.T, self.dim = X_true.shape

	def stretching(self, n, dn0, dn1, ns, ws=12, we=12, maxs=0.04):
	    """Return stretched segment.
	    Parameters
	    ----------
	    n : int
	        Total number of points.
	    dn0 : float
	        Initial grid spacing.
	    dn1 : float
	        Final grid spacing.
	    ns : int
	        Number of grid points with spacing equal to dn0
	    ws : int, optional
	        Number of grid points from stretching zero to stretching maxs
	    we : int, optional
	        Number of grid points from stretching maxs to stretching zero
	    maxs : float, optional
	        Maximum stretching (ds_i+1 - ds_i)/ds_i
	    Returns
	    -------
	    f: np.ndarray
	        One-dimensional np.array
	    """
	    ne = ns + np.log(dn1 / dn0) / np.log(1 + maxs)
	    s = np.array([maxs * 0.25 * (1 + erf(6 * (x - ns) / (ws))) * (1 - erf(6 * (x - ne) / we)) for x in range(n)])
	    f_ = np.empty(s.shape)
	    f_[0] = dn0
	    for k in range(1, len(f_)):
	        f_[k] = f_[k - 1] * (1 + s[k])
	    f = np.empty(s.shape)
	    f[0] = 0.0
	    for k in range(1, len(f)):
	        f[k] = f[k - 1] + f_[k]
	    return f

	def reshape_to_grid(self, tensor_phy):
	   
	    interp_func = interpolate.interp1d(np.arange(tensor_phy.shape[1]), tensor_phy, axis=1, kind='linear')

	    new_x_values = np.linspace(0, tensor_phy.shape[1]-1, 13100)
	    
	    grid = interp_func(new_x_values)
	    grid_reshaped = grid.reshape(tensor_phy.shape[0], 131, 100)
	    
	    return grid_reshaped

	def qr_decomposition(self, Psi_f) : 

		X_f = np.mean(Psi_f, axis = 0)
		dim = X_f.shape[-1]
		U, S, Vh = np.linalg.svd(X_f.T, full_matrices=False)
		U = U[:, :self.nobs]
		qr_idx = linalg.qr(U.T, pivoting=True)[-1]
		sensors_idx = qr_idx[:self.nobs]

		H = np.zeros((self.nobs, dim))
		for i, point in enumerate(sensors_idx) :
		 	H[i, point] = 1

		Y_obs = self.X_true @ H.T
		
		return sensors_idx, H, Y_obs



	def ks_pvalue_matrix(self, Psi_f):

		"""
		Compute the Kolmogorov–Smirnov p-value for each time step and
		each dimension of Psi_f with shape (E, T, D).

		Returns:
			pval_mat : array of shape (T, D)
			KS p-values for each (time, dimension).
		"""

		Psi_f = Psi_f[:, 100::300, :]

		E, T, D = Psi_f.shape
		pval_mat = np.zeros((T, D))

		for d in tqdm(range(D)):
			for t in range(T):

				samples = Psi_f[:, t, d]
				mu = samples.mean()
				sigma = samples.std(ddof=1)

				if sigma == 0:
					pval_mat[t, d] = 0.0
					continue

				cdf = lambda x: norm.cdf(x, loc=mu, scale=sigma)

				result = kstest(samples, cdf)
				pval_mat[t, d] = result.pvalue

		U_gaussianity = self.reshape_to_grid(pval_mat[:, :D//2])
		V_gaussianity = self.reshape_to_grid(pval_mat[:, D//2:])

		return U_gaussianity, V_gaussianity


	def plot_uv_gaussianity(self, U_gaussianity, V_gaussianity, threshold=0.05):
		"""
		pval_mat: (T, D) matrix of KS p-values
		X, Y: grid coordinates for pcolormesh
		threshold: significance level for Gaussianity (default 0.05)
		"""

		s1 = self.stretching(256, 0.033, 0.20, int(0.5/0.033+16), 16, 16, 0.04)
		s2 = self.stretching(128, 0.033, 0.20, int(0.5/0.033+16), 16, 16, 0.04)
		s = self.stretching(192, 0.033, 0.20, int(0.5/0.033+16), 16, 16, 0.04)
		x = np.r_[-s2[::-1], s1[1:]]
		y = np.r_[-s[::-1], s[1:]]
		a, b, c, d = -1.5, -2, 2, 8
		indices_x = np.where((x > a) & (x < d))[0]
		indices_y = np.where((y > b) & (y < c))[0]
		x_low, x_high = indices_x[0], indices_x[-1]
		y_low, y_high = indices_y[0], indices_y[-1]
		x = x[x_low : x_high]
		y = y[y_low : y_high]

		X, Y = np.meshgrid(x, y, indexing='ij')

		fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
		cmap = plt.cm.bwr
		col_U = axes[0].pcolormesh(
			X, Y, np.mean(U_gaussianity, axis = 0),
			cmap=cmap,
			shading='gouraud',
		)
		axes[0].set_title("U Gaussianity")
		fig.colorbar(col_U, ax=axes[0])

		col_V = axes[1].pcolormesh(
			X, Y, np.mean(V_gaussianity, axis = 0),
			cmap=cmap,
			shading='gouraud',
	
		)
		axes[1].set_title("V Gaussianity")
		fig.colorbar(col_V, ax=axes[1])

		plt.show()







