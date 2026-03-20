import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from time import time


class Train_Methodology():

    def time_evolution(self, initial_x_n, initial_x_seq, initial_Phi_n, ph_size, context):

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
        for t in range(ph_size):
            
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


    def train_test_loss(self, args,  mode = "Train", dataloader = None):
        '''
        One Step Prediction method
        Requires: dataloader, model, optimizer
        '''
        self.args = args

        if mode == "Train":
            dataloader = self.train_dataloader 
            self.model.train() 

            for param in self.model.transformer.parameters():
                param.requires_grad = False
                
        elif mode == "Test":
            dataloader = self.test_dataloader if dataloader != None else dataloader
            self.model.eval()
        else:
            print("mode can be Train or Test")
            return None


        num_batches = len(dataloader)
        total_loss, total_Autoencoder_Loss, total_TransEvo_Loss, total_StateEvo_Loss = 0,0,0,0
        total_koop_ptg, total_seqmodel_ptg = 0,0
        total_uq = 0
        
        for Phi_seq, Phi_nn_ph in dataloader:

            Phi_seq = Phi_seq.to(self.device)
            Phi_nn_ph = Phi_nn_ph.to(self.device)
 
            Phi_seq, context = torch.split(Phi_seq, [Phi_seq.shape[-1]-self.param_dim, self.param_dim], dim=-1)
            Phi_nn_ph, context_nn = torch.split(Phi_nn_ph, [Phi_nn_ph.shape[-1]-self.param_dim, self.param_dim], dim=-1)
            
            ph_size = Phi_nn_ph.shape[1] # pred_horizon size can vary depending on future steps available in data

            Phi_n   = torch.squeeze(Phi_seq[:,-1,...])  
            Phi_n   = Phi_n[None,...] if (Phi_n.ndim == self.state_ndim) else Phi_n #[bs statedim]
            Phi_n_ph = torch.cat((Phi_n[:,None,...], Phi_nn_ph[:,:-1,...]), 1)    #[bs ph_size statedim]
            
            ####### flattening batchsize seqlen / batchsize pred_horizon ######
            Phi_seq   = torch.flatten(Phi_seq, start_dim = 0, end_dim = 1) #[bs*seqlen, statedim]
            context_flatten   = torch.flatten(context, start_dim = 0, end_dim = 1)

            Phi_nn_ph = torch.flatten(Phi_nn_ph, start_dim = 0, end_dim = 1) #[bs*ph_size, statedim]
            context_nn   = torch.flatten(context_nn, start_dim = 0, end_dim = 1)
            ###### obtain observables ######

            x_seq, Phi_seq_hat, mu, log_var = self.model.autoencoder(Phi_seq, context_flatten)
            x_nn_ph , Phi_nn_hat_ph_nolatentevol, _, _ = self.model.autoencoder(Phi_nn_ph, context_nn)

            ###### reshaping tensors in desired form ######
            sd = (self.statedim,) if str(type(self.statedim)) == "<class 'int'>" else self.statedim
            
            Phi_nn_ph   = Phi_nn_ph.reshape(int(Phi_nn_ph.shape[0]/ph_size), ph_size, *sd) #[bs ph_size statedim]
            Phi_nn_hat_ph_nolatentevol = Phi_nn_hat_ph_nolatentevol.reshape(int(Phi_nn_hat_ph_nolatentevol.shape[0]/ph_size), ph_size, *sd) #[bs pred_horizon statedim]
            Phi_seq_hat = Phi_seq_hat.reshape(int(Phi_seq_hat.shape[0]/self.seq_len), self.seq_len, *sd) #[bs seqlen statedim]
            Phi_n_hat   = torch.squeeze(Phi_seq_hat[:, -1, :])
            Phi_n_hat   = Phi_n_hat[None,...] if (Phi_n_hat.ndim == self.state_ndim) else Phi_n_hat #[bs statedim]

            Phi_n_hat_ph = torch.cat((Phi_n_hat[:,None,...], Phi_nn_hat_ph_nolatentevol[:,:-1,...]), 1)  #obtaining decoded state tensor
             
            x_nn_ph  = x_nn_ph.reshape(int(x_nn_ph.shape[0]/ph_size), ph_size, self.num_obs) #[bs ph_size obsdim]
            x_seq = x_seq.reshape(int(x_seq.shape[0]/self.seq_len), self.seq_len, self.num_obs) #[bs seqlen obsdim]
            x_n   = torch.squeeze(x_seq[:,-1,:])   
            x_n   = x_n[None,...] if (x_n.ndim == 1) else x_n #[bs obsdim]
            x_seq = x_seq[:,:-1,:] #removing the current timestep from sequence. The sequence length is one less than input
            x_nn_hat_ph, Phi_nn_hat_ph = self.time_evolution(x_n, x_seq, Phi_n, ph_size, context)

            mseLoss      = nn.MSELoss()
            TransEvo_Loss = mseLoss(x_nn_hat_ph, x_nn_ph) + jac_loss
            KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            mseLoss      = nn.MSELoss()
            Autoencoder_Loss  = mseLoss(Phi_n_hat_ph, Phi_n_ph) + self.beta_VAE * KLD 
            StateEvo_Loss     = mseLoss(Phi_nn_hat_ph, Phi_nn_ph)
            
            loss = TransEvo_Loss + 100*Autoencoder_Loss + self.lambda_stateloss*StateEvo_Loss 
            uncertainty = torch.mean(0.5*log_var.exp())
      
            if mode == "Train":
                # self.optimizer.zero_grad()
                for param in self.model.parameters():
                    param.grad = None
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            total_TransEvo_Loss +=  TransEvo_Loss.item()
            total_Autoencoder_Loss += Autoencoder_Loss.item()
            total_StateEvo_Loss    += StateEvo_Loss.item()
            total_uq += uncertainty.item()

        if mode == "Train" : 
            self.scheduler.step()

        avg_loss             = total_loss / num_batches
        avg_TransEvo_Loss     = total_TransEvo_Loss / num_batches
        avg_Autoencoder_Loss = total_Autoencoder_Loss / num_batches
        avg_StateEvo_Loss    = total_StateEvo_Loss / num_batches
        avg_uq               = total_uq / num_batches

        current_lr = self.optimizer.param_groups[0]['lr']

        Ldict = {'avg_loss': avg_loss, 'avg_TransEvo_Loss': avg_TransEvo_Loss,'avg_Autoencoder_Loss': avg_Autoencoder_Loss, 'avg_StateEvo_Loss': avg_StateEvo_Loss,  'learning_rate' : current_lr, 'Uncertainty' : avg_uq} 

        return Ldict
    
################################################################################################################################################
    
    def training_loop(self, args):
        '''
        Requires:
        model, optimizer, train_dataloader, val_dataloader, device
        '''
        print("Device: ", self.device)
        print("Untrained Test\n--------")

        test_Ldict = self.train_test_loss(args= args, mode = "Test", dataloader = self.test_dataloader)

        print(f"Test Loss: {test_Ldict['avg_loss']:<{6}}, Transfo Loss : {test_Ldict['avg_TransEvo_Loss']:<{6}}, Auto : {test_Ldict['avg_Autoencoder_Loss']:<{6}}, StateEvo : {test_Ldict['avg_StateEvo_Loss']:<{6}}")

        # min train loss
        self.min_train_loss = 1000 
        self.min_test_loss  = 1000
        
        print(f"################## Starting Training - with {self.lambda_stateloss} StateEvo loss###############")
         
        for ix_epoch in range(self.load_epoch, self.load_epoch + self.nepochs):

            #start time
            start_time = time()
            
            train_Ldict = self.train_test_loss(args= args, mode = "Train", dataloader = None)
            test_Ldict  = self.train_test_loss(args= args, mode = "Test", dataloader = self.test_dataloader)
            
            #PRINTING AND SAVING DATA
            print(f"Epoch {ix_epoch} ")
            print(f"Train Loss: {train_Ldict['avg_loss']:<{6}}, KoopEvo : {train_Ldict['avg_TransEvo_Loss']:<{6}}, Auto : {train_Ldict['avg_Autoencoder_Loss']:<{6}}, StateEvo : {train_Ldict['avg_StateEvo_Loss']:<{6}},  Learning rate: {train_Ldict['learning_rate']}, Uncertainty : {train_Ldict['Uncertainty']:<{6}}")
            print(f"Test Loss: {test_Ldict['avg_loss']:<{6}}, KoopEvo : {test_Ldict['avg_TransEvo_Loss']:<{6}}, Auto : {test_Ldict['avg_Autoencoder_Loss']:<{6}}, StateEvo : {test_Ldict['avg_StateEvo_Loss']:<{6}}, Uncertainty : {test_Ldict['Uncertainty']:<{6}}")

            indentation = 0
            writeable_loss = {"epoch":str(ix_epoch).rjust(indentation),"Train_Loss":str(train_Ldict['avg_loss']).rjust(indentation), "Train_TransEvo_Loss":str(train_Ldict['avg_TransEvo_Loss']).rjust(indentation),\
                              "Train_Autoencoder_Loss":str(train_Ldict["avg_Autoencoder_Loss"]).rjust(indentation),"Uncertainty" : str(train_Ldict["Uncertainty"]).rjust(indentation),\
                              "Train_StateEvo_Loss":str(train_Ldict["avg_StateEvo_Loss"]).rjust(indentation),\
                              "Test_Loss":str(test_Ldict['avg_loss']).rjust(indentation), "Test_TransEvo_Loss":str(test_Ldict['avg_TransEvo_Loss']).rjust(indentation),\
                              "Test_Autoencoder_Loss":str(test_Ldict["avg_Autoencoder_Loss"]).rjust(indentation), "Test_StateEvo_Loss":str(test_Ldict["avg_StateEvo_Loss"]).rjust(indentation), 
                              "Uncertainty" : str(test_Ldict["Uncertainty"]).rjust(indentation)}
            
            self.log.writerow(writeable_loss)
            self.logf.flush()
            
            #saving Min Loss weights and optimizer state
            if self.min_test_loss > test_Ldict["avg_loss"]:
                self.min_test_loss = test_Ldict["avg_loss"]
                torch.save({
                    'epoch':ix_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict':self.optimizer.state_dict()
                    }, self.exp_dir+'/'+ self.exp_name+"/model_weights/min_test_loss")
            
            if self.min_train_loss > train_Ldict["avg_loss"]:
                self.min_train_loss = train_Ldict["avg_loss"]
                torch.save({
                    'epoch':ix_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict':self.optimizer.state_dict()
                    }, self.exp_dir+'/'+ self.exp_name+"/model_weights/min_train_loss")

            if (ix_epoch%self.nsave == 0):

                try:
                    self.plot_learning_curves()
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")

                torch.save({
                    'epoch':ix_epoch,
                    'model_state_dict': self.model.state_dict(),
                    }, self.exp_dir+'/'+ self.exp_name+"/model_weights/at_epoch{epoch}".format(epoch=ix_epoch))
                

            #ending time
            end_time = time()
            print("Epoch Time Taken: ", end_time - start_time)
        

        #saving final weights
        torch.save({
                    'epoch':ix_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict':self.optimizer.state_dict()
                    }, self.exp_dir+'/'+ self.exp_name+"/model_weights/at_epoch{epoch}".format(epoch=ix_epoch))
        
        self.logf.close()