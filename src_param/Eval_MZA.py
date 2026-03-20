
import torch
import torch.nn as nn
import pickle
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import statsmodels.api as sm
from scipy.stats import gaussian_kde
from src_param.MZA_Experiment import MZA_Experiment
from torch.utils.data import DataLoader

torch.manual_seed(99)

class Eval_MZA(MZA_Experiment):

    def __init__(self, exp_dir, exp_name):

        args = pickle.load(open(exp_dir + "/" + exp_name + "/args","rb"))
        #safety measure for new parameters added in model
            
        super().__init__(args)
        self.exp_dir = exp_dir
        self.exp_name = exp_name
            
##################################################################################################################
    def load_weights(self, epoch_num = 500, min_test_loss = False, min_train_loss = False):

        if min_test_loss:
            PATH = self.exp_dir+'/'+ self.exp_name+"/model_weights/min_test_loss".format(epoch=epoch_num)

        elif min_train_loss:
            PATH = self.exp_dir+'/'+ self.exp_name+"/model_weights/min_train_loss".format(epoch=epoch_num)

        else:
            PATH = self.exp_dir+'/'+ self.exp_name+"/model_weights/at_epoch{epoch}".format(epoch=epoch_num)
        
    
        checkpoint = torch.load(PATH)
        self.model.load_state_dict(checkpoint['model_state_dict'])


##################################################################################################################
    def predict_multistep(self, initial_conditions, timesteps, context):

            '''
            Input
            -----
            initial_conditions (torch tensor): [num_trajs, statedim]
            context (torch.tensor): [1, D]
            timesteps (int): Number timesteps for prediction

            Returns
            x (torch tensor): [num_trajs timesteps obsdim] observable vetcor
            Phi (torch tensor): [num_trajs timesteps statedim] state vector
            '''

            self.model.eval()

            Phi_n  = initial_conditions  
            _, _, x_n, log_var = self.model.autoencoder(Phi_n, context)    #[num_trajs obsdim]
            x   = x_n[None,...].to("cpu")                    #[timesteps num_trajs obsdim]
            Phi = Phi_n[None, ...].to("cpu")                    #[timesteps num_trajs statedim]
   

            for n in range(timesteps):

                non_time_dims = (1,)*(x.ndim-1)   #dims apart from timestep in tuple form (1,1,...)
                if n >= self.seq_len:
                    i_start = n - self.seq_len + 1
                    x_seq_n = x[i_start:(n+1), ...].to(self.device)
                elif n==0:
                    # padding = torch.zeros(x[0].repeat(self.seq_len - 1, *non_time_dims).shape).to(self.device)
                    padding = x[0].repeat(self.seq_len - 1, *non_time_dims).to(self.device)
                    x_seq_n = x[0:(n+1), ...].to(self.device)
                    x_seq_n = torch.cat((padding, x_seq_n), 0)
                else:
                    # padding = torch.zeros(x[0].repeat(self.seq_len - n, *non_time_dims).shape).to(self.device)
                    padding = x[0].repeat(self.seq_len - n, *non_time_dims).to(self.device)
                    x_seq_n = x[1:(n+1), ...].to(self.device)
                    x_seq_n = torch.cat((padding, x_seq_n), 0)
                
                x_seq_n = torch.movedim(x_seq_n, 1, 0) #[num_trajs seq_len obsdim]
                x_seq_n = x_seq_n[:,:-1,:]

                x_nn     = self.model.transformer(x_seq_n, context.unsqueeze(1))
                Phi_nn = self.model.autoencoder.recover(x_nn, context)

                x   = torch.cat((x,x_nn[None,...].detach().cpu()), 0)
                Phi = torch.cat((Phi,Phi_nn[None,...].detach().cpu()), 0)

            x      = torch.movedim(x, 1, 0)   #[num_trajs timesteps obsdim]
            Phi    = torch.movedim(Phi, 1, 0) #[num_trajs timesteps statedim]

            return x, Phi

    def get_latent_dynamics(self, phi_test, context) : 

        self.model.eval()
       
        x_n, mu, log_var = self.model.autoencoder.encode(phi_test, context)

        return mu, log_var


    def variational_UQ_scale(self, phi_test, context, ens) : 
        '''
        Input
        -----
        phi_test (torch tensor): [T, statedim]
        context (torch.tensor): [1, param_dim]
        ens (int) : Size of the ensemble (N)

        Returns
        Phi (torch tensor): [ensemble_size timesteps statedim] state vector
        '''

        Phi_n_ens = []
        T, D = phi_test.shape
        context = torch.repeat_interleave(context, T, dim = 0)

        for i in range(ens) : 

            x_n, Phi_n, mu, log_var = self.model.autoencoder(phi_test, context)
            Phi_n_ens.append(Phi_n)

        Phi_n_ens = torch.stack(Phi_n_ens, dim = 0)

        return Phi_n_ens

    def ensemble_forecast(self, initial_condition, context, timesteps, seq_len, ens) : 

        '''
        Input
        -----
        initial_conditions (torch tensor): [1, statedim]
        context (torch.tensor): [1, param_dim]
        timesteps (int): Number timesteps for prediction
        seq_len (int) : Look-back window size
        ens (int) : Size of the ensemble (N)

        Returns
        Phi (torch tensor): [ensemble_size timesteps statedim] state vector
        '''

        ens_traj = []

        for j in range(ens):
            
            Phi_tp  = torch.repeat_interleave(initial_condition, seq_len, dim=0)
            context_VAE = torch.repeat_interleave(context, seq_len, dim = 0)
            print(f"Ensemble {j+1}")

            traj = []

            for t in range(timesteps):
                x, mu, log_var = self.model.autoencoder.encode(Phi_tp, context_VAE)
                x = x[:-1, :].unsqueeze(0)
                context_latent = context_VAE[:-1, :].unsqueeze(0)
                x_tp1 = self.model.transformer(x, context)
                Phi_tp1 = self.model.autoencoder.recover(x_tp1, context_latent[:, -1, :])    
                traj.append(Phi_tp1[0,:])
                Phi_tp = torch.cat([Phi_tp, Phi_tp1], dim=0)[-seq_len:, :]
                
            ens_traj.append(torch.stack(traj))

            del Phi_tp, traj, x, mu, log_var, x_tp1, Phi_tp1
            torch.cuda.empty_cache()

        ens_traj = torch.stack(ens_traj)  # [ens, time, statedim]

        return ens_traj


    def forecast(self, initial_conditions, timesteps, context):

        '''
        Input
        -----
        initial_conditions (torch tensor): [t_in, statedim]
        context (torch.tensor): [1, 1]
        timesteps (int): Number of timesteps for prediction

        Returns
        x (torch tensor): [num_trajs timesteps obsdim] observable vetcor
        Phi (torch tensor): [num_trajs timesteps statedim] state vector
        '''
        t_in = initial_conditions.shape[0]
        if t_in < self.seq_len:
            # repeat first row
            first_val = initial_conditions[0].unsqueeze(0)        # [1, dim]
            padding = first_val.repeat(self.seq_len - t_in, 1)    # [seq_len - t_in, dim]
            Phi_in = torch.cat((padding, initial_conditions), dim=0)

        elif t_in == self.seq_len:
            Phi_in = initial_conditions

        else:  # t_in > seq_len
            Phi_in = initial_conditions[-self.seq_len:, :] 


        self.model.eval()
 
        m = Phi_in.shape[0]
        context_init = context.repeat(m, 1)
        x_n, mu_n, log_var = self.model.autoencoder.encode(Phi_in, context_init)    #x : [t_in obsdim]

        for n in range(timesteps - 1):

            x_in = mu_n[:-1, :].unsqueeze(0)
            # print(x_in.shape, context.shape)
            x_nn   = self.model.transformer(x_in, context.unsqueeze(0)) #[1 obsdim]
            x_n  = torch.cat((x_n,x_nn), 0)
            mu_n = x_n[-self.seq_len:, :]
        
        m = x_n.shape[0]
        context_final = context.repeat(m, 1)
        # print(x_n.shape, context_final.shape)
        Phi    = self.model.autoencoder.recover(x_n, context_final)

        if t_in < self.seq_len : 
            return Phi[self.seq_len - t_in:, :], x_n[self.seq_len - t_in:, :]
        else : 
            return Phi, x_n


    def plot_learning_curves(self):

        df = pd.read_csv(self.exp_dir+'/'+self.exp_name+"/out_log/log")

        min_trainloss = df.loc[df['Train_Loss'].idxmin(), 'epoch']
        print("Epoch with Minimum train_error: ", min_trainloss)

        min_testloss = df.loc[df['Test_Loss'].idxmin(), 'epoch']
        print("Epoch with Minimum test_error: ", min_testloss)

        #Total Loss
        plt.figure()
        plt.semilogy(df['epoch'],df['Train_Loss'], label="Train Loss")
        plt.semilogy(df['epoch'], df['Test_Loss'], label="Test Loss")
        plt.legend()
        plt.xlabel("Epochs")
        plt.savefig(self.exp_dir+'/'+self.exp_name+"/out_log/TotalLoss.png", dpi = 256, facecolor = 'w', bbox_inches='tight')

        #KoopEvo Loss
        plt.figure()
        plt.semilogy(df['epoch'],df['Train_TransEvo_Loss'], label="Train TransEvo Loss")
        plt.semilogy(df['epoch'], df['Test_TransEvo_Loss'], label="Test TransEvo Loss")
        plt.legend()
        plt.xlabel("Epochs")
        # plt.savefig(self.exp_dir+'/'+self.exp_name+"/out_log/AutoencoderLoss.png", dpi = 256, facecolor = 'w', bbox_inches='tight')

        #Autoencoder Loss
        plt.figure()
        plt.semilogy(df['epoch'],df['Train_Autoencoder_Loss'], label="Train Autoencoder Loss")
        plt.semilogy(df['epoch'], df['Test_Autoencoder_Loss'], label="Test Autoencoder Loss")
        plt.legend()
        plt.xlabel("Epochs")
        plt.savefig(self.exp_dir+'/'+self.exp_name+"/out_log/AutoencoderLoss.png", dpi = 256, facecolor = 'w', bbox_inches='tight')

        #State Loss
        plt.figure()
        plt.semilogy(df['epoch'],df['Train_StateEvo_Loss'], label="Train State Evolution Loss")
        plt.semilogy(df['epoch'], df['Test_StateEvo_Loss'], label="Test State Evolution Loss")
        plt.legend()
        plt.xlabel("Epochs")
        plt.savefig(self.exp_dir+'/'+self.exp_name+"/out_log/StateLoss.png", dpi = 256, facecolor = 'w', bbox_inches='tight')

        # #UQ
        # plt.figure()
        # plt.plot(df['epoch'],df['Uncertainty'], label="Train State Mean Uncertainty")
        # plt.plot(df['epoch'], df['Uncertainty'], label="Test State Mean Uncertainty")
        # plt.legend()
        # plt.xlabel("Epochs")
        # plt.savefig(self.exp_dir+'/'+self.exp_name+"/out_log/StateLoss.png", dpi = 256, facecolor = 'w', bbox_inches='tight')

    ###########################################################################
