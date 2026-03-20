import numpy as np 
import torch
from src_param.Eval_MZA import Eval_MZA
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from scipy.special import erf
from scipy import interpolate, linalg
from scipy.stats import energy_distance, wasserstein_distance
from tqdm import tqdm 
import random 

class Eval : 

	def __init__(self, model, X_true_Re, selected_U, selected_V, Re, Re_train, ens, points, all_re_evaluation) : 

		self.model = model 
		self.X_true_Re = X_true_Re
		self.selected_U = selected_U
		self.selected_V = selected_V
		self.Re = Re
		self.Re_train = Re_train
		self.ens = ens
		self.points = points
		self.p, self.T, self.dim = X_true_Re.shape
		self.all_re_evaluation = all_re_evaluation

		if torch.cuda.is_available():
			self.device = torch.device("cuda")
		elif torch.backends.mps.is_available():
			self.device = torch.device("mps") 
		else:
			self.device = torch.device("cpu")

		self.Re_arr = [80,90,100,110,120,130,140]
		self.index_re = self.Re_arr.index(self.Re)

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

	def plot_selected_points(self, X, Y, selected, UQ, field, title) : 

		"""
		To plot  figure with observation locations
		Input : 
		selected : np.array coordinates of points selected as observations
		"""

		if field == 'U': 
			style = {'marker': 'o', 's': 50, 'facecolors': 'white', 'edgecolors': 'black'}

		elif field == 'V':
			style = {'marker': 's', 's': 50, 'facecolors': 'white', 'edgecolors': 'black'}

		fig, ax = plt.subplots(figsize=(10, 4)) 
		colormap = plt.pcolormesh(X, Y, UQ, cmap='bwr', shading='gouraud')

		for i, point in enumerate(selected):
		    x_point = (point*4) // 100
		    y_point = (point*4) % 100
		    ax.scatter(X[x_point, y_point], Y[x_point, y_point], **style)
		    ax.scatter(0, 0, color = 'white', s = 4000)
		# plt.title(title)
		plt.colorbar(colormap, ax=ax)

	def evaluate_manifold(self) : 

		self.model.model.eval()
		Phi_in = torch.tensor(self.X_true_Re[self.index_re], dtype = torch.float32).to(self.device)
		context = torch.tensor([self.Re/140], dtype = torch.float32).to(self.device)
		context = context.unsqueeze(0)
		context= context.repeat(self.T, 1)
		x_n, Phi_out, mu, log_var = self.model.model.autoencoder(Phi_in, context)
		Phi_out = Phi_out.detach().cpu().numpy()
		l2_err = (Phi_out - self.X_true_Re[self.index_re])**2/np.mean(self.X_true_Re[self.index_re])

		x = self.X_true_Re[self.index_re]
		y = Phi_out
		abs_diff = np.abs(x - y)
		sq_diff = (x - y)**2
		rel_l1_total = np.sum(abs_diff) / (np.sum(np.abs(x)) + 1e-12) * 100
		rel_l2_total = np.sqrt(np.sum(sq_diff)) / (np.sqrt(np.sum(x**2)) + 1e-12) * 100

		return rel_l1_total, rel_l2_total


	def wasserstein_dist(self, signal1, signal2):
	    """
	    Compute Wasserstein distance dimension-wise between two signals of shape [T, D],
	    then average over the D dimensions to return a scalar.
	    
	    Args:
	        signal1, signal2: np.arrays of shape [T, D]
	        
	    Returns:
	        scalar average Wasserstein distance over D dimensions
	    """
	    T, D = signal1.shape
	    distances = []

	    for d in range(D) :

	        s1 = signal1[:, d]
	        s2 = signal2[:, d]

	        dist = energy_distance(s1,s2)
	        distances.append(dist)

	    return np.mean(distances)

	def plot_wasserstein_distance(self) : 

		self.model.model.eval()
		errors = []
		uqs = []

		for i in tqdm(range(self.p)) : 
			initial_condition = self.X_true_Re[i, 0:1, :]
			initial_condition = torch.tensor(initial_condition, dtype = torch.float32).to(self.device)
			context = torch.tensor([self.Re_arr[i]/140], dtype = torch.float32).to(self.device)
			context = context.unsqueeze(0)
			X_f_torch, latent_f = self.model.forecast(initial_condition, self.T, context)
			Psi_f_torch = self.model.variational_UQ_scale(X_f_torch, context, self.ens)
			Psi_f = Psi_f_torch.detach().cpu().numpy()
			X_f = X_f_torch.detach().cpu().numpy() #np.mean(Psi_f, axis = 0)
			Var_f = np.std(Psi_f, axis = 0)

			inferred_U = X_f[:, :self.dim//2] 
			inferred_V = X_f[:, self.dim//2:] 
			inferred_k = 0.5 * (inferred_U**2 + inferred_V**2)

			U = self.X_true_Re[i, :, :self.dim//2]
			V = self.X_true_Re[i, :, self.dim//2:]
			k = 0.5 * (U**2 + V**2)
		
			errors.append(self.wasserstein_dist(k, inferred_k))
			uqs.append(np.mean(Var_f))

		# print(errors,uqs)
		rounded_errors = [round(val, 6) for val in errors]
		rounded_uqs = [round(val, 6) for val in uqs]
		print(rounded_errors, rounded_uqs)

		indices = np.array(self.Re_arr)
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))  # Create two subplots side by side

		ax1.plot(self.Re_arr, errors, alpha=1, color='darkgrey', linewidth=3)
		ax1.scatter(self.Re_arr, errors, color='cornflowerblue', alpha=0.8, label='2-W-distance', marker='^', s=120)
		for idx, Re in enumerate(self.Re_train):
		    if idx == 0:
		        ax1.axvline(Re, color='black', linestyle='--', alpha=1, linewidth=2, label='Training sample')
		    else:
		        ax1.axvline(Re, color='black', linestyle='--', alpha=1, linewidth=2)
		ax1.axvline(self.Re, color='red', linestyle='--', alpha=1, linewidth=2, label='VAE retraining sample (DA)')
		ax1.set_xlabel('Reynolds Number', fontsize=20)
		ax1.set_ylabel('Energy distance', fontsize=20)
		ax1.set_xticks(self.Re_arr)
		ax1.set_ylim(0, 0.12)
		ax1.set_xticklabels([str(i) for i in self.Re_arr], rotation=45)
		ax1.grid(True, linestyle=':', alpha=0.6)
		ax1.legend(loc='lower right', fontsize=12)

		ax2.plot(self.Re_arr, uqs, alpha=1, color='darkgrey', linewidth=3)
		ax2.scatter(self.Re_arr, uqs, color='deeppink', alpha=0.8, label='Uncertainty Quantification', marker='^', s=120)
		for idx, Re in enumerate(self.Re_train):
		    if idx == 0:
		        ax2.axvline(Re, color='black', linestyle='--', alpha=1, linewidth=2, label='Training sample')
		    else:
		        ax2.axvline(Re, color='black', linestyle='--', alpha=1, linewidth=2)
		ax2.axvline(self.Re, color='red', linestyle='--', alpha=1, linewidth=2, label='VAE retraining sample (DA)')
		ax2.set_xlabel('Reynolds Number', fontsize=20)
		ax2.set_ylabel('Uncertainty Quantification', fontsize=20)
		ax2.set_xticks(self.Re_arr)
		ax2.set_ylim(0.0035,0.0050)
		ax2.set_xticklabels([str(i) for i in self.Re_arr], rotation=45)
		ax2.grid(True, linestyle=':', alpha=0.6)
		ax2.legend(loc='lower right', fontsize=12)

		plt.tight_layout()

	def plot_U_V_signals(self, X, points):
	    """
	    Generates a list of 9 locations in the flow domain and plots 
	    U(t) and V(t) at these locations.

	    Parameters:
	        X (np.ndarray): flow field [T, dim]
	        points (np.ndarray) : locations across the flow field [n]
	    """

	    # Split predicted fields
	    U = X[:, :self.dim//2]                   # shape [T, D]
	    V = X[:, self.dim//2:]                  # shape [T, D]

	    # Split reference/true fields
	    U_true = self.X_true_Re[self.index_re, :, :self.dim//2]
	    V_true = self.X_true_Re[self.index_re, :, self.dim//2:]  

	    D = self.dim // 2
	    # points = random.sample(range(0, D), 9)  # indices 0..D-1

	    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
	    axes = axes.flatten()

	    for i, idx in enumerate(points):
	        ax = axes[i]

	        # Plot U and U_true
	        ax.plot(U[:, idx], label=f'U pred (x={idx})', color='cornflowerblue', linewidth=1)
	        ax.plot(U_true[:, idx], label=f'U true (x={idx})', color='cornflowerblue', linewidth = 2, linestyle='--')

	        # Plot V and V_true
	        ax.plot(V[:, idx], label=f'V pred (x={idx})', color='coral', linewidth=1)
	        ax.plot(V_true[:, idx], label=f'V true (x={idx})', color='coral', linewidth = 2, linestyle='--')

	        ax.set_title(f"Point {idx}")
	        ax.set_xlabel("t")
	        ax.set_ylabel("value")
	        ax.grid(True)

	        # Only show legend on the first subplot
	        if i == 0:
	            ax.legend()

	    plt.tight_layout()

	def Psi_f(self, initial_condition) : 

		"""
		Uses the object 'model' to generate the forecast ensemble over tn snapshots. 
		Parameters : 

		initial_condition (np.ndarray) : Initialization buffer [1, dim] 
        
		returns : 
		Psi_f (np.ndarray) : Forecast ensemble [N, tn, dim] or if possible [N, tn+seq_len, dim]
		X_f (np.ndarray) : Mean of the forecast ensemble [tn, dim] or [tn+seq_len, dim]
		"""

		self.model.model.eval()
		initial_condition = torch.tensor(initial_condition, dtype = torch.float32).to(self.device)
		context = torch.tensor([self.Re/140], dtype = torch.float32).to(self.device)
		context = context.unsqueeze(0)
		X_f_torch, latent_f = self.model.forecast(initial_condition, self.T, context)
		Psi_f_torch = self.model.variational_UQ_scale(X_f_torch, context, self.ens)
		Psi_f = Psi_f_torch.detach().cpu().numpy()
		X_f = X_f_torch.detach().cpu().numpy()

		return Psi_f, X_f

	def kinetic_energy(self, U, V) : 
		
		return np.mean(0.5*(U**2 + V**2), axis = -1) 

	def plots(self, init_err_l2_U, init_err_l2_V, init_err_l1_U, init_err_l1_V, plot) : 

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

		initial_condition = self.X_true_Re[self.index_re, 0:1, :]
		Psi_f, X_f = self.Psi_f(initial_condition)
		U_ens = Psi_f[:, :, :self.dim//2]
		V_ens = Psi_f[:, :, self.dim//2:]

		U_true = self.X_true_Re[self.index_re, :, :self.dim//2]
		V_true = self.X_true_Re[self.index_re, :, self.dim//2:]

		U_var = np.var(U_ens, axis = 0)
		V_var = np.var(V_ens, axis = 0)
		U_mean = np.mean(U_ens, axis = 0)
		V_mean = np.mean(V_ens, axis = 0)

		U_var_grid = self.reshape_to_grid(U_var)
		V_var_grid = self.reshape_to_grid(V_var)
		U_mean_grid = self.reshape_to_grid(U_mean)
		V_mean_grid = self.reshape_to_grid(V_mean)

		U_true_grid = self.reshape_to_grid(U_true)
		V_true_grid = self.reshape_to_grid(V_true)

		if plot == True : 

			self.plot_selected_points(X, Y, self.selected_U, np.mean(U_var_grid, axis = 0), 'U', 'Mean UQ-U and sensors')
			self.plot_selected_points(X, Y, self.selected_V, np.mean(V_var_grid, axis = 0), 'V', 'Mean UQ-V and sensors')
			self.plot_U_V_signals(X_f, self.points)

		k_forecast = self.kinetic_energy(U_mean, V_mean)
		k_true = self.kinetic_energy(U_true, V_true)

		Error_U_l2 = (U_mean-U_true)**2
		Error_V_l2 = (V_mean-V_true)**2
		Error_U_l1 = np.abs(U_mean-U_true)
		Error_V_l1 = np.abs(V_mean-V_true)

		mean_Error_U_l2 = np.mean(Error_U_l2)
		mean_Error_V_l2 = np.mean(Error_V_l2)
		mean_Error_U_l1 = np.mean(Error_U_l1)
		mean_Error_V_l1 = np.mean(Error_V_l1)

		relative_Error_U_l2 = (mean_Error_U_l2 / np.mean(U_true ** 2)) * 100
		relative_Error_V_l2 = (mean_Error_V_l2 / np.mean(V_true ** 2)) * 100
		relative_Error_U_l1 = (mean_Error_U_l1 / np.mean(np.abs(U_true))) * 100
		relative_Error_V_l1 = (mean_Error_V_l1 / np.mean(np.abs(V_true))) * 100

		print(f"Error U (L2): {mean_Error_U_l2:.4f} (Relative: {relative_Error_U_l2:.1f}%)")
		print(f"Error V (L2): {mean_Error_V_l2:.4f} (Relative: {relative_Error_V_l2:.1f}%)")
		print(f"Error U (L1): {mean_Error_U_l1:.4f} (Relative: {relative_Error_U_l1:.1f}%)")
		print(f"Error V (L1): {mean_Error_V_l1:.4f} (Relative: {relative_Error_V_l1:.1f}%)")

		l1, l2 = self.evaluate_manifold()
		print(f"Error manifold l2 relative  {l2:.2f}%")
		print(f"Error manifold l1 relative  {l1:.2f}%")

		timewise_err_U_l2 = np.mean(Error_U_l2, axis = -1)
		timewise_err_V_l2 = np.mean(Error_V_l2, axis = -1)
		timewise_err_U_l1 = np.mean(Error_U_l1, axis = -1)
		timewise_err_V_l1 = np.mean(Error_V_l1, axis = -1)

		if plot and all(var is not None for var in [init_err_l2_U, init_err_l2_V, init_err_l1_U, init_err_l1_V]): 

			time = np.arange(self.T)

			fig, axes = plt.subplots(2, 2, figsize=(10, 6))

			# --- L2 Errors ---
			# U (L2)
			axes[0, 0].plot(time, timewise_err_U_l2, label='U (L2 - retraining)', color='cornflowerblue', linewidth=2)
			axes[0, 0].plot(time, np.mean(init_err_l2_U, -1), label='U (L2)', color='coral', linewidth=2)
			axes[0, 0].set_xlabel('Time')
			axes[0, 0].set_ylabel('L2 Error')
			axes[0, 0].set_title('U - Timewise L2 Error')
			axes[0, 0].legend()
			axes[0, 0].grid(True)

			# V (L2)
			axes[0, 1].plot(time, timewise_err_V_l2, label='V (L2 - retraining)', color='cornflowerblue', linewidth=2)
			axes[0, 1].plot(time, np.mean(init_err_l2_V, -1), label='V (L2)', color='coral', linewidth=2)
			axes[0, 1].set_xlabel('Time')
			axes[0, 1].set_ylabel('L2 Error')
			axes[0, 1].set_title('V - Timewise L2 Error')
			axes[0, 1].legend()
			axes[0, 1].grid(True)

			# --- L1 Errors ---
			# U (L1)
			axes[1, 0].plot(time, timewise_err_U_l1, label='U (L1 - retraining)', color='cornflowerblue', linewidth=2)
			axes[1, 0].plot(time, np.mean(init_err_l1_U, -1), label='U (L1)', color='coral', linewidth=2)
			axes[1, 0].set_xlabel('Time')
			axes[1, 0].set_ylabel('L1 Error')
			axes[1, 0].set_title('U - Timewise L1 Error')
			axes[1, 0].legend()
			axes[1, 0].grid(True)

			# V (L1)
			axes[1, 1].plot(time, timewise_err_V_l1, label='V (L1 - retraining)', color='cornflowerblue', linewidth=2)
			axes[1, 1].plot(time, np.mean(init_err_l1_V, -1), label='V (L1)', color='coral', linewidth=2)
			axes[1, 1].set_xlabel('Time')
			axes[1, 1].set_ylabel('L1 Error')
			axes[1, 1].set_title('V - Timewise L1 Error')
			axes[1, 1].legend()
			axes[1, 1].grid(True)

			plt.tight_layout()  

			plt.figure(figsize=(10, 5))
			plt.plot(time*0.0024, k_forecast, label='Forecast Kinetic Energy', color='brown', linewidth = 2)
			plt.plot(time*0.0024, k_true, label='True Kinetic Energy', color='black', linewidth = 2, linestyle = '--')
			# plt.xlabel('Time')
			# plt.ylabel('Kinetic Energy')
			# plt.title('Kinetic Energy')
			# plt.legend()
			plt.grid(False)
			
			err_U_space_l2_grid = np.mean(self.reshape_to_grid(Error_U_l2), axis = 0)
			err_V_space_l2_grid = np.mean(self.reshape_to_grid(Error_V_l2), axis = 0)
			err_U_space_l2_init_grid = np.mean(self.reshape_to_grid(init_err_l2_U), axis = 0)
			err_V_space_l2_init_grid = np.mean(self.reshape_to_grid(init_err_l2_V), axis = 0)

			# Compute global min and max for consistent scaling
			all_fields = [
			    err_U_space_l2_grid,
			    err_V_space_l2_grid,
			    err_U_space_l2_init_grid,
			    err_V_space_l2_init_grid
			]

			global_min = min(field.min() for field in all_fields)
			global_max = max(field.max() for field in all_fields)

			# Create a 2x2 subplot grid
			fig, axes = plt.subplots(2, 2, figsize=(10, 6))

			# --- L2 Errors ---
			# U (L2 - retraining)
			p1 = axes[0, 0].pcolormesh(X, Y, err_U_space_l2_grid, cmap='bwr', shading='gouraud', vmin=global_min, vmax=global_max)
			axes[0, 0].set_title('U - L2 Error (Retraining)')
			fig.colorbar(p1, ax=axes[0, 0])

			# V (L2 - retraining)
			p2 = axes[0, 1].pcolormesh(X, Y, err_V_space_l2_grid, cmap='bwr', shading='gouraud', vmin=global_min, vmax=global_max)
			axes[0, 1].set_title('V - L2 Error (Retraining)')
			fig.colorbar(p2, ax=axes[0, 1])

			# --- L2 Errors (Initial) ---
			# U (L2 - initial)
			p3 = axes[1, 0].pcolormesh(X, Y, err_U_space_l2_init_grid, cmap='bwr', shading='gouraud', vmin=global_min, vmax=global_max)
			axes[1, 0].set_title('U - L2 Error (Initial)')
			fig.colorbar(p3, ax=axes[1, 0])

			# V (L2 - initial)
			p4 = axes[1, 1].pcolormesh(X, Y, err_V_space_l2_init_grid, cmap='bwr', shading='gouraud', vmin=global_min, vmax=global_max)
			axes[1, 1].set_title('V - L2 Error (Initial)')
			fig.colorbar(p4, ax=axes[1, 1])

			# Add labels for rows and columns
			for ax, row_label in zip(axes[:, 0], ['L2 Error - Retraining', 'L2 Error - Initial']):
			    ax.set_ylabel(row_label, fontsize=12)

			for ax, col_label in zip(axes[0, :], ['U', 'V']):
			    ax.set_xlabel(col_label, fontsize=12)

			plt.tight_layout()


		if self.all_re_evaluation == True : 
			
			self.plot_wasserstein_distance()

		plt.show()

		return Error_U_l2, Error_V_l2, Error_U_l1, Error_V_l1 


# error = 0.02264, 0.01149, 0.019507, 0.021773, 0.011522, 0.020878, 0.030687 
# UQ = 0.004442, 0.004048, 0.004102, 0.004103, 0.004012, 0.004192, 0.004367