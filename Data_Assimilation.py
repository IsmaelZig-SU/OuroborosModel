import numpy as np 
from tqdm import tqdm
from scipy import interpolate, linalg
import matplotlib.pyplot as plt 
from scipy.special import erf
from scipy import interpolate, linalg

class Data_Assimilation :

	def __init__(self, Psi_f, Y_obs, downsampler, nobs, X_true, plots): 

		"""
		Parameters :
			Psi_f (np.ndarray): forecast ensemble [N, T, dim] 
			Y_obs (np.ndarray): Sparsed observations [T, n_obs]
			downsampler (np.ndarray): sensor's placement [n_obs, dim]
			nobs : (scalar) number of sensors 
			X_true (np.ndarray) : Only used at evaluation, True system states [T,dim]

		Returns : 
			Psi_a (np.ndarray): analysis ensemble [N,T,dim] 
	
		"""

		self.Psi_f = Psi_f 
		self.Y_obs = Y_obs
		self.H = downsampler 
		self.nobs = nobs 
		self.X_true = X_true
		self.plots = plots
		self.N, self.T, self.dim = self.Psi_f.shape

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

	def get_PfHt(self, ensemble, H):

	    """
	    Generate the product of the error covariance matrix from an ensemble of predictions
	    and the Transpose of the subsampling operator : Pf@H.T.
	    
	    Parameters:
	        ensemble (np.ndarray): Ensemble array of shape [N, dim],
	                               where N is the ensemble size and dim is the state dimension.
	        H (np.ndarray) : Downsampling operator [n_obs, dim]
	    
	    Returns:
	        PfHt (np.ndarray): Covariance projector matrix of shape [dim, nobs]
	    """

	    N, dim = ensemble.shape
	    mean = np.mean(ensemble, axis=0)
	    anomalies = ensemble - mean  # Shape: [N, dim]

	    H_ens = ensemble @ H.T
	    H_mean = np.mean(H_ens, axis = 0)
	    H_anomalies = H_ens - H_mean #Can also use H_ens - H_mean -> Difference in PhFt using Y_obs and H_mean is O 1e-12 
	   
	    PfHt = 1/(N - 1) * anomalies.T @ H_anomalies
	    
	    return PfHt

	def get_X_a_ens_t(self, Y_obs_t, PfHt, HPHT, Psi_t_ens, H, R) : 

		"""
		Generate the analysis ensemble Psi_a_t_ens from sparsed observations Y_obs and with the 
		loosy ensemble forecast Psi_t_ens at time t. 
	    
		Parameters:

		Y_obs_t (np.ndarray): Observations at time t [n_obs],
		PfHT (np.ndarray): Covariance projector at time t [dim, n_obs]          
		HPHT (np.ndarray): Covariance projected to observation space at time t [n_obs, n_obs]
		Psi_t_ens (np.ndarray): Original forecast ensemble at time t [N, dim]
		H (np.ndarray) : Downsampling operator [n_obs, dim]	   
		R (np.ndarray) : Diagonal noise matrix [n_obs, n_obs] 

		Returns:

		Psi_a_t_ens np.ndarray: Analysis ensemble at time t [N, dim]
		"""
		
		N, dim = Psi_t_ens.shape
		Psi_a_t_ens = np.zeros_like(Psi_t_ens)
		K = PfHt @ np.linalg.inv(HPHT + R)

		for ens in range(N) :

			X_ft = Psi_t_ens[ens, :]
			innovation = Y_obs_t - H @ X_ft
			update = K @ innovation.T
			X_at = X_ft +  update.T
			Psi_a_t_ens[ens,:] = X_at

		return Psi_a_t_ens

	def dynamical_rollout(self, Psi_f, Y_obs, H, nobs, obs_noise, X_true, verbose) :


		"""
		Iterates over time t to assimilate data sequentially
		Parameters : 

		Psi_f (np.ndarray): forecast ensemble [N, T, dim] 
		Y_obs (np.ndarray): Sparsed observations [T, n_obs]
		H (np.ndarray): Downsampling matrix [n_obs, dim]	
		nobs (int) : Number of observations
		obs_noise (float): observation noise (variance) 
		X_true (np.ndarray) : Only used at evaluation, True system states [T,dim]

		returns 

		Psi_a (np.ndarray) : Analysis ensemble [N, T, dim]
		"""

		N, T, dim = Psi_f.shape 
		X_f = np.mean(Psi_f, axis = 0) #Mean of the forecast ensmeble [T,D]
		error_init = np.mean((X_f - X_true)**2)

		U_f = X_f[..., :dim//2]
		V_f = X_f[..., dim//2:]
		U_true = X_true[..., :dim//2]
		V_true = X_true[..., dim//2:]
		error_init_U = np.mean((U_f - U_true)**2)
		error_init_V = np.mean((V_f - V_true)**2)	

		if verbose == True : 
			print(f"Original Psi-MSE: {error_init:.4f}")
			print(f"Original U-MSE: {error_init_U:.4f}")
			print(f"Original V-MSE: {error_init_V:.4f}")

		R = np.eye(nobs)*obs_noise
		Psi_a = np.zeros_like(Psi_f)

		error_arr = []
		error_init_arr = []
		improvement_arr = []


		for t in tqdm(range(T), desc="Processing"):

			Psi_f_t = Psi_f[:, t, :] #[N, dim]
			Y_obs_t = Y_obs[t, :] #[dim, n_obs]
			PfHt = self.get_PfHt(Psi_f_t, H) #[dim, nobs]
			HPHT = H @ PfHt #[nobs, nobs]

			Psi_a_t_ens = self.get_X_a_ens_t(Y_obs_t, PfHt, HPHT, Psi_f_t, H, R)
			Psi_a[:, t, :] = Psi_a_t_ens

		X_a = np.mean(Psi_a, axis = 0) #Mean of the analysis ensmeble [T,D]
		U_a = X_a[..., :dim//2]
		V_a = X_a[..., dim//2:]

		error_final = np.mean((X_a - X_true)**2)
		error_final_U = np.mean((U_a - U_true)**2)
		error_final_V = np.mean((V_a - V_true)**2)	

		if verbose == True : 

			print(f"Final Psi-MSE: {error_final:.4f}")
			print(f"Final U-MSE: {error_final_U:.4f}")
			print(f"Final V-MSE: {error_final_V:.4f}")

			if error_init > 1e-7 : 
				improvement  = (error_init - error_final)/error_init

			else : 
				improvement = 1

			if error_init_U > 1e-7 : 
				improvement_U  = (error_init_U - error_final_U)/error_init_U

			else : 
				improvement_U = 1

			if error_init_V > 1e-7 : 
				improvement_V  = (error_init_V - error_final_V)/error_init_V

			else : 
				improvement_V = 1

			print(f"Improvement: {improvement*100:.1f}%")
			print(f"Improvement U: {improvement_U*100:.1f}%")
			print(f"Improvement V: {improvement_V*100:.1f}%")

			time = np.arange(self.T)*0.0024

			fig, axes = plt.subplots(1, 2, figsize=(14, 5))

			if self.plots == True :
		
				axes[0].plot(time, np.mean((U_f - U_true)**2, axis =-1)/np.mean(U_true**2, axis = -1)*100, label='U forecast', color='cornflowerblue', linewidth=2)
				axes[0].plot(time, np.mean((U_a - U_true)**2, axis =-1)/np.mean(U_true**2, axis = -1)*100, label='U analysis', color='coral', linewidth=2)
				axes[0].set_xlabel('Time', fontname = 'Times New Roman', fontsize = 18)
				axes[0].set_ylabel('U - L2 Error [%]', fontname = 'Times New Roman', fontsize = 18)
				#axes[0].set_title('U - Timewise L2 Error')
				# axes[0, 0].legend()
				axes[0].grid(True)

				axes[1].plot(time, np.mean((V_f - V_true)**2, axis =-1)/np.mean(V_true**2, axis = -1)*100, label='V forecast', color='cornflowerblue', linewidth=2)
				axes[1].plot(time, np.mean((V_a - V_true)**2, axis =-1)/np.mean(V_true**2, axis = -1)*100, label='V analysis', color='coral', linewidth=2)
				axes[1].set_xlabel('Time', fontname = 'Times New Roman', fontsize = 18)
				axes[1].set_ylabel('V - L2 Error [%]', fontname = 'Times New Roman', fontsize = 18)
				# axes[1].set_title('V - Timewise L2 Error', fontsize = 18)
				# axes[0, 0].legend()
				axes[1].grid(True)
				plt.show()

				err_U_f = self.reshape_to_grid((U_f - U_true)**2/np.mean(U_true**2))*100
				err_V_f = self.reshape_to_grid((V_f - V_true)**2/np.mean(V_true**2))*100
				err_U_a = self.reshape_to_grid((U_a - U_true)**2/np.mean(U_true**2))*100
				err_V_a = self.reshape_to_grid((V_a - V_true)**2/np.mean(V_true**2))*100

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

				fig, axes = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)
				cmap = plt.cm.bwr

				# --- Compute vmin/vmax for U-row ---
				U_f = np.mean(err_U_f, axis=0)
				U_a = np.mean(err_U_a, axis=0)

				vmin_U = min(U_f.min(), U_a.min())
				vmax_U = max(U_f.max(), U_a.max())

				# --- Compute vmin/vmax for V-row ---
				V_f = np.mean(err_V_f, axis=0)
				V_a = np.mean(err_V_a, axis=0)

				vmin_V = min(V_f.min(), V_a.min())
				vmax_V = max(V_f.max(), V_a.max())

				# ========== Plot U row ==========
				col_U_f = axes[0, 0].pcolormesh(
				    X, Y, U_f, cmap=cmap, shading='gouraud',
				    vmin=vmin_U, vmax=vmax_U
				)
				axes[0, 0].scatter(0, 0, color = 'white', s = 4000)
				axes[0, 0].set_title("U (forecast) - L2 Error", fontname='Times New Roman', fontsize=20)

				col_U_a = axes[0, 1].pcolormesh(
				    X, Y, U_a, cmap=cmap, shading='gouraud',
				    vmin=vmin_U, vmax=vmax_U
				)
				axes[0, 1].scatter(0, 0, color = 'white', s = 4000)
				axes[0, 1].set_title("U (analysis) - L2 Error", fontname='Times New Roman', fontsize=20)

				# --- One colorbar for U row ---
				cbar_U = fig.colorbar(col_U_f, ax=axes[0, :], shrink=0.95, location='right')
				cbar_U.set_label("Error (%)", fontsize=18)

				# ========== Plot V row ==========
				col_V_f = axes[1, 0].pcolormesh(
				    X, Y, V_f, cmap=cmap, shading='gouraud',
				    vmin=vmin_V, vmax=vmax_V
				)
				axes[1, 0].scatter(0, 0, color = 'white', s = 4000)
				axes[1, 0].set_title("V (forecast) - L2 Error", fontname='Times New Roman', fontsize=20)

				col_V_a = axes[1, 1].pcolormesh(
				    X, Y, V_a, cmap=cmap, shading='gouraud',
				    vmin=vmin_V, vmax=vmax_V
				)
				axes[1, 1].scatter(0, 0, color = 'white', s = 4000)
				axes[1, 1].set_title("V (analysis) - L2 Error", fontname='Times New Roman', fontsize=20)

				# --- One colorbar for V row ---
				cbar_V = fig.colorbar(col_V_f, ax=axes[1, :], shrink=0.95, location='right')
				cbar_V.set_label("Error (%)", fontsize=18)

				plt.show()



		return Psi_a


