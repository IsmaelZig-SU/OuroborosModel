import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from time import time
import numpy as np 

class Train_Methodology :

    def __init__(self, model, param_dim, dataloader, pred_horizon, seq_len, statedim, num_obs, beta_VAE, device, epochs, optimizer, exp_name, Re) : 

        self.model = model
        self.param_dim = param_dim
        self.dataloader = dataloader
        self.ph_size = pred_horizon
        self.seq_len = seq_len 
        self.statedim = statedim
        self.num_obs = num_obs 
        self.beta_VAE = beta_VAE
        self.device = device 
        self.epochs = epochs 
        self.optimizer = optimizer
        self.exp_name = exp_name
        self.exp_dir = 'pre_trained_model/'
        self.Re = Re

        self.state_ndim = 1 


    def time_evolution(self, initial_x_n, initial_x_seq, initial_Phi_n, context):

        """
        Calculates multistep prediction from koopman and seqmodel while training
        Inputs
        ------
        initial_x_n (torch tensor): [bs obsdim]
        initial_x_seq (torch tensor): [bs seq_len obsdim]
        initial_Phi_n (torch tensor): [bs statedim]
        ph_size (int) : variable pred_horizon acccording to future data available
        context (torch tensor) : [bs, seq_len, 1]
    
        Returns
        -------
        x_nn_hat_ph (torch_tensor): [bs pred_horizon obsdim]
        Phi_nn_hat (torch_tensor): [bs pred_horizon statedim]
        """
        x_n   = initial_x_n 
        x_seq = initial_x_seq

        trans_out_ph = x_n.clone()[:,None,...]   #[bs 1 obsdim]
        Phi_nn_hat_ph = initial_Phi_n.clone()[:,None,...] #[bs 1 statedim]
        x_nn_hat_ph = x_n.clone()[:,None,...]
        Phi_nn_hat_ph = initial_Phi_n.clone()[:,None,...] #[bs 1 statedim]

        #Evolving in Time
        for t in range(self.ph_size):
            
            context_vae = context[:,-1, :]
            context_transfo = context_vae.unsqueeze(1)
            trans_out = self.model.transformer(x_seq, context_transfo)

            if t==0 : 
                trans_out_ph[:,0,...] = trans_out

            else : 
                trans_out_ph = torch.cat((trans_out_ph, trans_out[:, None, ...]), 1)

            x_nn_hat = trans_out
            Phi_nn_hat = self.model.autoencoder.recover(trans_out, context_vae)

            x_nn_hat_ph   = torch.cat((x_nn_hat_ph,x_nn_hat[:,None,...]), 1)
            Phi_nn_hat_ph = torch.cat((Phi_nn_hat_ph,Phi_nn_hat[:,None,...]), 1)

            x_seq = torch.cat((x_seq[:,1:,...],x_n[:,None,...]), 1)
            x_n = x_nn_hat

        return x_nn_hat_ph[:,1:,...], Phi_nn_hat_ph[:,1:,...]


################################################################################################################################################

    def train_loss(self):
 
        self.model.train() 
        
        # print("Freezing Transformer parameters")
        for param in self.model.transformer.parameters():
            param.requires_grad = False

        num_batches = len(self.dataloader)
        total_loss, total_Autoencoder_Loss, total_TransEvo_Loss, total_StateEvo_Loss = 0,0,0,0
        total_uq = 0
        
        for Phi_seq, Phi_nn_ph in self.dataloader:

            Phi_seq = Phi_seq.to(self.device)
            Phi_nn_ph = Phi_nn_ph.to(self.device)
 
            Phi_seq, context = torch.split(Phi_seq, [Phi_seq.shape[-1]-self.param_dim, self.param_dim], dim=-1)
            Phi_nn_ph, context_nn = torch.split(Phi_nn_ph, [Phi_nn_ph.shape[-1]-self.param_dim, self.param_dim], dim=-1)

            Phi_n   = torch.squeeze(Phi_seq[:,-1,...])  
            Phi_n   = Phi_n[None,...] if (Phi_n.ndim == self.state_ndim) else Phi_n #[bs statedim]
            Phi_n_ph = torch.cat((Phi_n[:,None,...], Phi_nn_ph[:,:-1,...]), 1)    #[bs ph_size statedim]
            
            ####### flattening batchsize seqlen / batchsize pred_horizon ######
            Phi_seq   = torch.flatten(Phi_seq, start_dim = 0, end_dim = 1)
            context_flatten   = torch.flatten(context, start_dim = 0, end_dim = 1)

            Phi_nn_ph = torch.flatten(Phi_nn_ph, start_dim = 0, end_dim = 1) 
            context_nn   = torch.flatten(context_nn, start_dim = 0, end_dim = 1)
            ###### obtain observables ######

            x_seq, Phi_seq_hat, mu, log_var = self.model.autoencoder(Phi_seq, context_flatten)
            x_nn_ph , Phi_nn_hat_ph_nolatentevol, _, _ = self.model.autoencoder(Phi_nn_ph, context_nn)

            ###### reshaping tensors in desired form ######
            sd = (self.statedim,) if str(type(self.statedim)) == "<class 'int'>" else self.statedim
            
            Phi_nn_ph   = Phi_nn_ph.reshape(int(Phi_nn_ph.shape[0]/self.ph_size), self.ph_size, *sd) #[bs ph_size statedim]
            Phi_nn_hat_ph_nolatentevol = Phi_nn_hat_ph_nolatentevol.reshape(int(Phi_nn_hat_ph_nolatentevol.shape[0]/self.ph_size), self.ph_size, *sd) #[bs pred_horizon statedim]
            Phi_seq_hat = Phi_seq_hat.reshape(int(Phi_seq_hat.shape[0]/self.seq_len), self.seq_len, *sd) #[bs seqlen statedim]
            Phi_n_hat   = torch.squeeze(Phi_seq_hat[:, -1, :])
            Phi_n_hat   = Phi_n_hat[None,...] if (Phi_n_hat.ndim == self.state_ndim) else Phi_n_hat #[bs statedim]

            Phi_n_hat_ph = torch.cat((Phi_n_hat[:,None,...], Phi_nn_hat_ph_nolatentevol[:,:-1,...]), 1)  #obtaining decoded state tensor
             
            x_nn_ph  = x_nn_ph.reshape(int(x_nn_ph.shape[0]/self.ph_size), self.ph_size, self.num_obs) #[bs ph_size obsdim]
            x_seq = x_seq.reshape(int(x_seq.shape[0]/self.seq_len), self.seq_len, self.num_obs) #[bs seqlen obsdim]
            x_n   = torch.squeeze(x_seq[:,-1,:])   
            x_n   = x_n[None,...] if (x_n.ndim == 1) else x_n #[bs obsdim]
            x_seq = x_seq[:,:-1,:] #removing the current timestep from sequence. The sequence length is one less than input
            Re_batch = context[:, 0, 0].detach().cpu().numpy()*140
            count = np.sum(Re_batch == self.Re)
            x_nn_hat_ph, Phi_nn_hat_ph = self.time_evolution(x_n, x_seq, Phi_n, context)

            mseLoss      = nn.MSELoss()
            TransEvo_Loss = mseLoss(x_nn_hat_ph, x_nn_ph)
            KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            mseLoss      = nn.MSELoss()
            Autoencoder_Loss  = mseLoss(Phi_n_hat_ph, Phi_n_ph) + self.beta_VAE * KLD 
            StateEvo_Loss     = mseLoss(Phi_nn_hat_ph, Phi_nn_ph)
            loss = TransEvo_Loss + StateEvo_Loss + 100*Autoencoder_Loss
            uncertainty = torch.mean(log_var.exp())
    
            for param in self.model.parameters():
                param.grad = None
            loss.backward()
            self.optimizer.step()  

            total_loss += loss.item()
            total_TransEvo_Loss +=  TransEvo_Loss.item()
            total_Autoencoder_Loss += Autoencoder_Loss.item()
            total_StateEvo_Loss    += StateEvo_Loss.item()
            total_uq += uncertainty.item()

        avg_loss             = total_loss / num_batches
        avg_TransEvo_Loss     = total_TransEvo_Loss / num_batches
        avg_Autoencoder_Loss = total_Autoencoder_Loss / num_batches
        avg_StateEvo_Loss    = total_StateEvo_Loss / num_batches
        avg_uq               = total_uq / num_batches


        Ldict = {'avg_loss': avg_loss, 'avg_TransEvo_Loss': avg_TransEvo_Loss,'avg_Autoencoder_Loss': avg_Autoencoder_Loss, 'avg_StateEvo_Loss': avg_StateEvo_Loss, 'Uncertainty' : avg_uq} 

        return Ldict
    
################################################################################################################################################
    
    def training_loop(self):
        
        # min train loss
        min_train_loss = 1000 
        
        print(f"################## Starting Re-Training ###############")
         
        for ix_epoch in range(self.epochs):

            #start time
            start_time = time()
            train_Ldict = self.train_loss()
    
            print(f"Epoch {ix_epoch} ")
            print(f"Train Loss: {train_Ldict['avg_loss']:<{6}}, Transfomer : {train_Ldict['avg_TransEvo_Loss']:<{6}}, Autoencoder : {train_Ldict['avg_Autoencoder_Loss']:<{6}}, StateEvo : {train_Ldict['avg_StateEvo_Loss']:<{6}}")
            
            # if min_train_loss > train_Ldict["avg_loss"]:
            #     min_train_loss = train_Ldict["avg_loss"]
            #     torch.save({
            #         'epoch':ix_epoch,
            #         'model_state_dict': self.model.state_dict(),
            #         'optimizer_state_dict':self.optimizer.state_dict()
            #         }, self.exp_dir+'/'+ self.exp_name+"/model_weights/min_train_loss")

            end_time = time()
            print("Epoch Time Taken: ", end_time - start_time)
        

        # #saving final weights
        # torch.save({
        #             'epoch':ix_epoch,
        #             'model_state_dict': self.model.state_dict(),
        #             'optimizer_state_dict':self.optimizer.state_dict()
        #             }, self.exp_dir+'/'+ self.exp_name+"/model_weights/at_epoch{epoch}".format(epoch=ix_epoch))