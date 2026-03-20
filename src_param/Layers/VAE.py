import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
import os 

class Autoencoder(nn.Module):
 
    def __init__(self, args, model_eval = False):
        super(Autoencoder, self).__init__()
        
        self.args = args
        print(f"AE_Model: VAE network, {self.args['num_obs']} observables, KLD regularization (beta) : {self.args['beta_VAE']}")
    
 
        if not model_eval:

            self.input_size  = self.args["statedim"]
            self.latent_size = self.args["num_obs"]
            self.linear_ae   = self.args["linear_autoencoder"]
            self.context_dim = self.args['param_dim']
 
            #encoder layers
            self.e_fc1 = nn.Linear(self.input_size+self.context_dim, 512)
            self.e_fc2 = nn.Linear(512, 256)
            self.e_fc3 = nn.Linear(256, 128)
            self.e_fc4 = nn.Linear(128, 64)
            self.e_fc5 = nn.Linear(64, self.latent_size)

            self.mu = nn.Linear(self.latent_size, self.latent_size)
            self.log_var = nn.Linear(self.latent_size, self.latent_size)
 
            #decoder layers
            self.d_fc1 = nn.Linear(self.latent_size + self.context_dim, 64)
            self.d_fc2 = nn.Linear(64, 128)
            self.d_fc3 = nn.Linear(128, 256)
            self.d_fc4 = nn.Linear(256, 512)
            self.d_fc5 = nn.Linear(512, self.input_size)

            self.relu    = nn.ReLU()
            # Freeze all layers except the bottleneck
            # self.freeze_except_bottleneck()

    # def freeze_except_bottleneck(self):

    #     for name, param in self.named_parameters():
    #         if name in ['d_fc2.weight', 'd_fc2.bias', 'e_fc4.weight', 'e_fc4.bias']:

    #             param.requires_grad = True
    #         else:

    #             print("freezing : ", name)
    #             param.requires_grad = False
 
    def encoder(self, x):
          
        x = self.relu(self.e_fc1(x))
        x = self.relu(self.e_fc2(x))
        x = self.relu(self.e_fc3(x))
        x = self.relu(self.e_fc4(x))
        x = self.e_fc5(x)
        
        return x
    
    def decoder(self, x):
 
        #non linear encoder
        x = self.relu(self.d_fc1(x))
        x = self.relu(self.d_fc2(x))
        x = self.relu(self.d_fc3(x))
        x = self.relu(self.d_fc4(x))
        x = self.d_fc5(x)
 
        return x

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
 
    def forward(self, Phi_n, context):
      
        Phi_in = torch.cat([Phi_n, context], axis=-1)
        encoded = self.encoder(Phi_in)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        x_n = self.reparameterize(mu, log_var)
        x_in = torch.cat([x_n, context], axis=-1)
        Phi_n_hat = self.decoder(x_in)
 
        return x_n, Phi_n_hat, mu, log_var
 
    def recover(self, x_n, context):
        
        x_in = torch.cat([x_n, context], axis=-1)
        Phi_n_hat = self.decoder(x_in)
        return Phi_n_hat

    def encode(self, Phi_n, context):

        Phi_in = torch.cat([Phi_n, context], axis=-1)
        encoded = self.encoder(Phi_in)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        x_n = self.reparameterize(mu, log_var)

        return x_n, mu, log_var

    def encode_variational(self, Phi_n, context, ens_size):

        Phi_in = torch.cat([Phi_n, context], axis=-1)
        encoded = self.encoder(Phi_in)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        x_n_ens = []
        for i in range(ens_size) : 
            x_n = self.reparameterize(mu, log_var)
            x_n_ens.append(x_n)
        x_n_ens = torch.stack(x_n_ens, dim = 0)

        return x_n_ens, mu, log_var

    def _num_parameters(self):
        count = 0
        for name, param in self.named_parameters():
            # print(name, param.numel())
            count += param.numel()
        return count

    def get_weights(self):

        """
        Returns a dictionary of weights
        """
        all_weights = {'e_fc1' : self.e_fc1.weight.detach().cpu().numpy(), 'e_fc2' : self.e_fc2.weight.detach().cpu().numpy(),
                        'e_fc3' : self.e_fc3.weight.detach().cpu().numpy(), 'e_fc4' : self.e_fc4.weight.detach().cpu().numpy(), 
                        'e_fc5' : self.e_fc5.weight.detach().cpu().numpy(), 'mu' : self.mu.weight.detach().cpu().numpy(), 
                        'log_var' : self.log_var.weight.detach().cpu().numpy()}
        all_biases = {'e_fc1' : self.e_fc1.bias.detach().cpu().numpy(), 'e_fc2' : self.e_fc2.bias.detach().cpu().numpy(),
                        'e_fc3' : self.e_fc3.bias.detach().cpu().numpy(), 'e_fc4' : self.e_fc4.bias.detach().cpu().numpy(), 
                        'e_fc5' : self.e_fc5.bias.detach().cpu().numpy(), 'mu' : self.mu.bias.detach().cpu().numpy(), 
                        'log_var' : self.log_var.bias.detach().cpu().numpy()}
        return all_weights, all_biases
