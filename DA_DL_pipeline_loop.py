import sys
import os
import torch
import numpy as np 
from Data_Assimilation import Data_Assimilation
from initialisation import Initialisation
from src_param.Eval_MZA import Eval_MZA
from data_loader import SequenceForecastDataset
from retrain_methodology import Train_Methodology
from torch.utils.data import DataLoader
from Evaluation_DA_DL import Eval
import matplotlib.pyplot as plt 
from scipy.signal import savgol_filter
import random
import argparse

# plt.rc("text", usetex=True)
# plt.rcParams.update(
#     {
#         'font.family': 'serif',
#         'mathtext.fontset': 'cm',
#         'mathtext.rm': 'lmodern',
#         'font.size': 20,
#         'legend.fontsize': 16.5,
#         'xtick.labelsize': 16.5,
#         'ytick.labelsize': 16.5,
#         'axes.titlesize': 18
#     }
# )

class Retrain_DA : 

    """
    retraining and DA loop 
    Parameters : 
    model (object class MZA_Experiment) : Autoregressive model 
    t_init (np.ndarray) : Initialization buffer [1, dim] or [seq_len, dim]
    epsilon (float) : convergence criterion between assimilated ensemble and forecast ensemble
    param_dim (int) : dimension of the parameter space (If only Re, then 1)
    Re (float) : Reynolds number (parametetric variable)
    ens (int) : ensemble cardinality (N)
    seq_len (int) : Lookback windiow size 
    pred_horizon (int) : Forecast training window
    latent_dim (int) : dimension of the latent space 
    beta_VAE : KLD regularization in VAE loss 
    tn (int) : Timewise slice for the main loop 
    nobs (int) : Number of sensors (sparsed observations)
    H (np.ndarray) : Downsampling matrix (sensor locations) [nobs, dim]
    X_true (np.ndarray) : True states (only used at evaluation):  [T, dim]
    obs_noise (float) : observation noise (variance)
    epochs (int) : Retraining epochs
    iter_max : (int) : Maximum number of DA+retrain loops 
    exp_name : (str) : name of the model's folder on Trained_experiment
    """

    def __init__(self, model, t_init, epsilon, batch_size, param_dim, Re, ens, seq_len, pred_horizon, 
        latent_dim, beta_VAE, tn, nobs, H, X_true, obs_noise, epochs, iter_max, exp_name, plot_energy, DA_plots) : 

        self.model = model 
        self.t_init = t_init
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.Re = Re
        self.param_dim = param_dim
        self.ens = ens
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon
        self.num_obs = latent_dim
        self.beta_VAE = beta_VAE
        self.tn = tn 
        self.nobs = nobs 
        self.H = H
        self.X_true = X_true 
        self.obs_noise = obs_noise
        self.epochs = epochs
        self.iter_max = iter_max
        self.exp_name = exp_name
        self.plot_energy = plot_energy
        self.DA_plots = DA_plots

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps") 
        else:
            self.device = torch.device("cpu")

        self.T, self.dim = X_true.shape

    def Psi_f(self, t_init) : 

        """
        Uses the object 'model' to generate the forecast ensemble over tn snapshots. 
        Parameters : 

        t_init (np.ndarray) : Initialization buffer [1, dim] or [seq_len, dim]
        
        returns : 
        Psi_f (np.ndarray) : Forecast ensemble [N, tn, dim] or if possible [N, tn+seq_len, dim]
        X_f (np.ndarray) : Mean of the forecast ensemble [tn, dim] or [tn+seq_len, dim]
        """

        self.model.model.eval()
        initial_condition = torch.tensor(t_init, dtype = torch.float32).to(self.device)
        context = torch.tensor([self.Re/140], dtype = torch.float32).to(self.device)
        context = context.unsqueeze(0)
        X_f_torch, latent_f = self.model.forecast(initial_condition, self.tn, context)
        Psi_f_torch = self.model.variational_UQ_scale(X_f_torch, context, self.ens)
        Psi_f = Psi_f_torch.detach().cpu().numpy()
        # X_f = X_f_torch.detach().cpu().numpy()
        X_f = np.mean(Psi_f, axis = 0)

        return Psi_f, X_f

    def savgol_smooth(self, signal, polyorder=4):

        """
        Apply Savitzky–Golay filtering timewise to a 2D signal of shape [T, dim].

        Parameters:
            signal (np.ndarray): 2D input signal of shape [T, dim].
            polyorder (int): Order of the polynomial used to fit the samples.
                Default is 4.

        Returns:
            np.ndarray: Smoothed signal of the same shape as the input.
        """

        window_length = self.T  

        if window_length % 2 == 0:
            window_length -= 1

        smoothed_signal = np.zeros_like(signal)
        for d in range(self.dim):
            smoothed_signal[:, d] = savgol_filter(
                signal[:, d],
                window_length=window_length,
                polyorder=polyorder
            )

        return smoothed_signal

    def energy(self, X) : 

        """
        Returns kinetic energy of slice for evaluation (optional) 

        Parameters : 
        X (np.ndarray) : velocity components U and V [T, dim]

        Returns : 
        Kinetic energy of flow field [T] 
        """

        U = X[:, :self.dim//2]
        V = X[:, self.dim//2:]
        k = 0.5*(U**2 + V**2)

        return np.mean(k, axis = -1)

    def retrain_methology(self, training_data, Y_obs) :

        """
        Retraining loop every tn timewise slice. 
        Parameters : 

            training_data (np.ndarray) : Original training data over P parameters : [P, tn, dim]
                    if T//tn > 1 and t > seq_len training_data : [P, tn + seq_len, dim] 
            Y_obs (np.ndarray) : Observations [tn, nobs] or, similarly [tn + seq_len, nobs]
                    It is important that all snapshots across arrays are consistent

        Returns : 

            X_f (np.ndarray) : Forecast ensemble 

        """
        criterions = []
        for iteration in range(self.iter_max) : 

            Psi_f, X_f = self.Psi_f(self.t_init)
          
            if self.plot_energy == True and iteration == 0 : 

                k_f = self.energy(X_f)
                k_true = self.energy(self.X_true)
                plt.figure(figsize=(10, 5))
                plt.plot(k_f, label = 'forecast', linewidth = 2, color = 'coral')
                plt.plot(k_true, label = 'true', linewidth = 2, color = 'cornflowerblue')
                plt.title('Before')
                plt.legend()

            da = Data_Assimilation(Psi_f, Y_obs, self.H, self.nobs, self.X_true, self.DA_plots)
            Psi_analysis = da.dynamical_rollout(Psi_f, Y_obs, self.H, self.nobs, self.obs_noise, self.X_true, verbose=True) 
            X_a = np.mean(Psi_analysis, axis = 0)
            
            criterion = np.mean((X_a- X_f)**2)
            print(f"Iteration {iteration+1}, residual : {criterion:.4f}")
            criterions.append(criterion)
            if  criterion < self.epsilon : 
                print(f"Convergence: {criterion:.4f}")
                if self.plot_energy == True : 

                    plt.figure(figsize=(10, 5))
                    plt.plot(criterion, color = 'plum', linewidth = 2)
                    plt.title('L2 distance analysis - forecast')
                    plt.show()

                break 

            else : 

                context = np.full((self.T, 1), self.Re/140)
                X_a = np.concatenate([X_a, context], axis = 1)
                X_a = X_a[np.newaxis, ...]
                # print(training_data.shape, X_a.shape)
                retraining_data = np.concatenate([training_data, X_a], axis = 0)
                #retraining_data = training_data
                retrain_dataset  = SequenceForecastDataset(retraining_data, self.seq_len, self.pred_horizon)

                loader = DataLoader(
                    retrain_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=0)
                
                optimizer = torch.optim.Adam(self.model.model.parameters(), lr = 1e-5, weight_decay=1e-5)
                training_loop = Train_Methodology(self.model.model, self.param_dim, loader, self.pred_horizon, self.seq_len, 
                    self.dim, self.num_obs, self.beta_VAE, self.device, self.epochs, optimizer, self.exp_name, self.Re)
                training_loop.training_loop()

            Psi_f, X_f = self.Psi_f(self.t_init)

            if self.plot_energy == True  and iteration == self.iter_max - 1 : 

                k_f = self.energy(X_f)
                k_a = self.energy(X_a[0, :, :-1])
                k_true = self.energy(self.X_true)
                plt.figure(figsize=(10, 5))
                plt.plot(k_f, label = 'forecast', linewidth = 2, color = 'coral')
                plt.plot(k_a, label = 'analysis', linewidth = 2, linestyle = '--', color = 'grey')
                plt.plot(k_true, label = 'true', linewidth = 2, color = 'cornflowerblue')
                plt.title('After')
                plt.legend()
                if len(criterions) > 1 : 

                    plt.figure(figsize=(10, 5))
                    plt.plot(criterions, color = 'plum', linewidth = 2)
                    plt.title('L2 distance analysis - forecast')
                
                plt.show()

        return X_f 

torch.cuda.empty_cache()

def get_args():
    """
    Defines the hyperparameter configuration and experiment settings.
    """
    parser = argparse.ArgumentParser(description="Deep Learning Data Assimilation for Fluid Dynamics")

    # Physical and Model Parameters
    parser.add_argument('--epsilon', type=float, default=1e-4, help='Observation')
    parser.add_argument('--param_dim', type=int, default=1, help='Parameter dimensionality')
    parser.add_argument('--Re', type=int, default=140, help='Reynolds number for evaluation')
    parser.add_argument('--ens', type=int, default=10, help='Ensemble size')
    parser.add_argument('--seq_len', type=int, default=9, help='Context sequence length')
    parser.add_argument('--pred_horizon', type=int, default=2, help='Prediction horizon')
    parser.add_argument('--latent_dim', type=int, default=4, help='VAE latent space dimensionality')
    parser.add_argument('--beta_VAE', type=float, default=3e-4, help='Beta weight for VAE loss')

    # Simulation and Training Settings
    parser.add_argument('--tn', type=int, default=1000, help='Retraining time window')
    parser.add_argument('--nobs', type=int, default=16, help='Number of observation sensors')
    parser.add_argument('--obs_noise', type=float, default=1e-8, help='Observation noise variance')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--iter_max', type=int, default=1, help='Maximum DA iterations')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')

    # Paths and Metadata
    parser.add_argument('--model_path', type=str, default='pre_trained_model/', help='Base model directory')
    parser.add_argument('--exp_name', type=str, default="test", help='Experiment identifier')
    parser.add_argument('--train_data_name', type=str, default='90120.npy', help='Filename for training data')
    parser.add_argument('--test_data_name', type=str, default='adaptive_sampling_test.npy', help='Filename for validation data')
    parser.add_argument('--folder_path', type=str, default='Data/', help='Base data directory')
    
    # Array/List Parameters
    parser.add_argument('--Re_train', type=list, default=[90, 120], help='Reynolds numbers used in training')
    parser.add_argument('--Re_arr', type=list, default=[80, 90, 100, 110, 120, 130, 140], help='Reynolds number range')

    # Flags
    parser.add_argument('--plot_energy', action='store_true', default=False)
    parser.add_argument('--all_re_evaluation', action='store_true', default=True)
    parser.add_argument('--DA_plots', action='store_true', default=False)

    return parser.parse_args()

def main():
    args = get_args()

    # 1. Data Ingestion
    print(f"--- [Status] Loading training data: {args.train_data_name} ---")
    train_dataset = np.load(args.folder_path + args.train_data_name)
    train_dataset = train_dataset[:, 0::2, :] # Temporal downsampling
    
    filename = "adaptive_sampling_test.npy"
    data_set = np.load(args.folder_path + filename)
    test_set = data_set[3:, 0::2, :]
    
    # Extract ground truth for specific Reynolds number
    X_true_Re = test_set[..., :-1]
    re_idx = args.Re_arr.index(args.Re)
    X_true_fs = X_true_Re[re_idx, :, :]
    T, dim = X_true_fs.shape

    # 2. Model Initialization
    print(f"--- [Status] Loading model: {args.exp_name} ---")
    model = Eval_MZA(args.model_path, args.exp_name)
    model.load_weights(min_test_loss=True)

    # 3. Sensor Placement and Initialization
    print("--- [Status] Identifying optimal sensor locations via QR Decomposition ---")
    init_engine = Initialisation(model, X_true_fs, args.nobs, args.Re, args.ens)
    Psi_f, X_f = init_engine.Psi_f()
    
    sensors_idx, H, Y_obs = init_engine.qr_decomposition(Psi_f)
    selected_U = np.array([i for i in sensors_idx if i < dim // 2])
    selected_V = np.array([i - dim // 2 for i in sensors_idx if i >= dim // 2])
    
    # Selection of stochastic points for visualization
    points = random.sample(range(0, dim // 2), 9)

    # 4. Preliminary Evaluation
    evaluation_engine = Eval(model, X_true_Re, selected_U, selected_V, args.Re, 
                             args.Re_train, args.ens, points, all_re_evaluation=False)
    err_metrics = evaluation_engine.plots(None, None, None, None, plot=False)
    err_l2_U, err_l2_V, err_l1_U, err_l1_V = err_metrics

    # 5. Iterative Data Assimilation Loop
    for step in range(T // args.tn):
        print(f"\n--- Processing Snapshot: {step * args.tn} to {(step + 1) * args.tn} ---")

        # Temporal windowing for assimilation
        start_idx = step * args.tn
        if start_idx > args.seq_len:
            window_start = start_idx - args.seq_len + 1
            training_subset = train_dataset[:, window_start:(step + 1) * args.tn, :]
            t_init = X_f[-args.seq_len:, :]
        else:
            window_start = start_idx
            training_subset = train_dataset[:, window_start:(step + 1) * args.tn, :]
            t_init = X_true_fs[0:1, :]

        Y_obs_window = X_true_fs[window_start:(step + 1) * args.tn, :] @ H.T
        X_true_window = X_true_fs[window_start:(step + 1) * args.tn, :]

        # Execute Retraining/Assimilation methodology
        da_module = Retrain_DA(
            model, t_init, args.epsilon, args.batch_size, args.param_dim, args.Re, 
            args.ens, args.seq_len, args.pred_horizon, args.latent_dim, args.beta_VAE,
            args.tn, args.nobs, H, X_true_window, args.obs_noise, args.epochs, 
            args.iter_max, args.exp_name, args.plot_energy, args.DA_plots
        )
        
        X_f = da_module.retrain_methology(training_subset, Y_obs_window)

    # 6. Final Performance Evaluation
    print("\n--- [Status] Finalizing evaluation and generating plots ---")
    final_eval = Eval(model, X_true_Re, selected_U, selected_V, args.Re, 
                      args.Re_train, args.ens, points, args.all_re_evaluation)
    final_eval.plots(err_l2_U, err_l2_V, err_l1_U, err_l1_V, plot=True)

if __name__ == "__main__":
    main()