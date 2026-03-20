import torch
import csv, pickle, copy

from src_param.Layers.MZANetwork import MZANetwork

from src_param.Train_Methods.Train_Methodology import Train_Methodology
from src_param.PreProc_Data.DynSystem_Data import DynSystem_Data
from torch.optim.lr_scheduler import StepLR, LambdaLR

import matplotlib.pyplot as plt
import pandas as pd
from src_param.utils.make_dir import mkdirs
torch.manual_seed(99)


class MZA_Experiment(DynSystem_Data, Train_Methodology):

    def __init__(self,args):

        #Device parameters
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps") 
        else:
            self.device = torch.device("cpu")
        
        #Data Parameters
        if str(type(args)) != "<class 'dict'>":

            self.dynsys      = args.dynsys
            self.train_size  = args.train_size
            self.batch_size  = args.bs
            self.ntransients = args.ntransients
            self.seq_len     = args.seq_len
            self.nenddata    = args.nenddata
            self.np          = args.noise_p
            self.param_dim   = args.nbr_ext_var

            #Autoncoder Parameters         
            self.autoencoder_model     = args.AE_Model 
            self.num_obs               = args.num_obs
            self.linear_autoencoder    = args.linear_autoencoder
            self.beta_VAE              = args.beta_VAE 

            #Transformer Parameters
            self.seq_model           = args.seq_model
            self.nattblocks          = args.nattblocks
            self.nheads              = args.nheads
            self.hidden_dim          = args.hidden_dim
            self.seq_model_weight    = args.seq_model_weight

            #Model Training # Model Hyper-parameters
            self.learning_rate          = args.lr      
            self.nepochs                = args.nepochs
            self.norm_input             = args.norm_input         #if input should be normalised
            self.pred_horizon           = args.pred_horizon
            self.lambda_stateloss       = args.lambda_stateloss

            #Directory Parameters
            self.nsave         = args.nsave                 #after how many epochs to save
            self.info          = args.info                  #extra info in the saved driectory name
            self.exp_dir       = args.exp_dir
            self.exp_name      ="sl{sl}_obs{numobs}_bs{bs}_attblks{attblocks}_atthds{nheads}_tr{tr}_ph{ph}_lbdaStateLoss{lambda_stateloss}_nhd{nhd}_{beta_VAE}_{info}".format(nhd = args.hidden_dim, beta_VAE = args.beta_VAE, sl = args.seq_len, numobs = args.num_obs, bs=args.bs, attblocks = args.nattblocks, nheads = args.nheads, tr = args.ntransients, ph = args.pred_horizon, lambda_stateloss = args.lambda_stateloss, info=args.info)
            self.data_dir      = args.data_dir
            self.no_save_model = args.no_save_model
            self.load_epoch    = args.load_epoch

            self.nsave         = args.nsave                 #after how many epochs to save
            self.info          = args.info                  #extra info in the saved driectory name
            self.exp_dir       = args.exp_dir

            if self.load_epoch != 0:
                self.exp_name = args.load_exp_name
                self.load_model = True
                print(self.exp_name)

            self.args = args
        
        else:
            for k, v in args.items():
                setattr(self, k, v)
        
        #printing out important information
        print("########## Imp Info ##########")
        print("System: ", self.dynsys)
        
        #emptying gpu cache memory
        torch.cuda.empty_cache()

###########################################################################################################################
    def make_directories(self):
        '''
        Makes Experiment Directory
        '''
        directories = [self.exp_dir,
                    self.exp_dir + '/' + self.exp_name,
                    self.exp_dir + '/' + self.exp_name + "/model_weights",
                    self.exp_dir + '/' + self.exp_name + "/out_log",
                    ]
        mkdirs(directories)
    
###########################################################################################################################
    def log_data(self, load_model):

        self.metrics = ["epoch","Train_Loss","Train_TransEvo_Loss","Train_Autoencoder_Loss","Train_StateEvo_Loss"\
                               ,"Test_Loss","Test_TransEvo_Loss","Test_Autoencoder_Loss","Test_StateEvo_Loss", "Uncertainty"]

        self.logf = open(self.exp_dir + '/' + self.exp_name + "/out_log/log", "w")
        self.log = csv.DictWriter(self.logf, self.metrics)
        self.log.writeheader()

        print("Logger Initialised")

###########################################################################################################################
    def save_args(self):

        #saving args
        with open(self.exp_dir+'/'+self.exp_name+"/args", 'wb') as f:
            args_dict = copy.deepcopy(self.__dict__)

            #deleting some high memory args
            print("\n", args_dict.keys(), "\n\n")
            del args_dict['lp_data']
            del args_dict['train_data']
            del args_dict['test_data']
            del args_dict['train_dataset']
            del args_dict['test_dataset']
            del args_dict['train_dataloader']
            del args_dict['test_dataloader']
            # #adding data_args
            pickle.dump(args_dict, f)
            print("Saved Args")

    
###########################################################################################################################


    def lr_lambda(self, epoch):

        if 0 <= self.load_epoch + epoch < 500:
            return 1e-4
        elif 500 <= self.load_epoch + epoch < 1000:
            return 5e-5
        elif 1000 <= self.load_epoch + epoch < 2000:
            return 1e-5
        elif 2000 <= self.load_epoch + epoch < 3000:
            return 5e-6
        elif 3000 <= self.load_epoch + epoch < 4000: 
            return 1e-6
        elif 4000 <= self.load_epoch + epoch < 6000: 
            return 5e-7
        else:
            return 1e-7


    def main_train(self, load_model):


        #Making Experiment Directory
        self.make_directories()

        #Loading and visualising data
        print("########## LOADING DATASET ##########")
        print("Data Dir: ", self.data_dir)
        self.load_and_preproc_data()

        # #Creating Statevariable Dataset
        self.create_dataset()

        #Creating Model
        print("########## SETTING UP MODEL ##########")
        if not load_model:
            self.model = MZANetwork(self.__dict__).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 1, weight_decay=1e-5)
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)
        
        if not load_model:
                
            #saving args
            self.save_args()
        
        if load_model:
            self.model = MZANetwork(self.__dict__).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 1, weight_decay=1e-5)
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)
            args = pickle.load(open(self.exp_dir + "/" + self.exp_name + "/args","rb"))
            PATH = self.exp_dir+'/'+ self.exp_name+"/model_weights/min_test_loss"
            checkpoint = torch.load(PATH)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Loading model parameters")

            
        # Initiating Data Logger
        self.log_data(load_model)

        #Training Model
        self.training_loop(self.args)

        #Saving Model
        if self.no_save_model:
            print("model saved in "+ self.exp_dir+'/'+self.exp_name+'/'+self.exp_name)

    
###########################################################################################################################
    def plot_learning_curves(self):

        df = pd.read_csv(self.exp_dir+'/'+self.exp_name+"/out_log/log")

        min_trainloss = df.loc[df['Train_Loss'].idxmin(), 'epoch']
        # print("Epoch with Minimum train_error: ", min_trainloss)

        #Total Loss
        plt.figure()
        plt.semilogy(df['epoch'],df['Train_Loss'], label="Train Loss")
        plt.semilogy(df['epoch'], df['Test_Loss'], label="Test Loss")
        plt.legend()
        plt.xlabel("Epochs")
        plt.savefig(self.exp_dir+'/'+self.exp_name+"/out_log/TotalLoss.png", dpi = 256, facecolor = 'w', bbox_inches='tight')
        plt.close()

        #Transfo Loss
        plt.figure()
        plt.semilogy(df['epoch'],df['Train_TransEvo_Loss'], label="Train Transformer Evo Loss")
        plt.semilogy(df['epoch'], df['Test_TransEvo_Loss'], label="Test Transformer Evo Loss")
        plt.legend()
        plt.xlabel("Epochs")
        plt.savefig(self.exp_dir+'/'+self.exp_name+"/out_log/TransEvo.png", dpi = 256, facecolor = 'w', bbox_inches='tight')
        plt.close()

        #Autoencoder Loss
        plt.figure()
        plt.semilogy(df['epoch'],df['Train_Autoencoder_Loss'], label="Train Autoencoder Loss")
        plt.semilogy(df['epoch'], df['Test_Autoencoder_Loss'], label="Test Autoencoder Loss")
        plt.legend()
        plt.xlabel("Epochs")
        plt.savefig(self.exp_dir+'/'+self.exp_name+"/out_log/AutoencoderLoss.png", dpi = 256, facecolor = 'w', bbox_inches='tight')
        plt.close()

        #State Loss
        plt.figure()
        plt.semilogy(df['epoch'],df['Train_StateEvo_Loss'], label="Train State Evolution Loss")
        plt.semilogy(df['epoch'], df['Test_StateEvo_Loss'], label="Test State Evolution Loss")
        plt.legend()
        plt.xlabel("Epochs")
        plt.savefig(self.exp_dir+'/'+self.exp_name+"/out_log/StateLoss.png", dpi = 256, facecolor = 'w', bbox_inches='tight')
        plt.close()

        #UQ
        plt.figure()
        plt.plot(df['epoch'],df['Uncertainty'], label="Train State Mean Uncertainty")
        plt.legend()
        plt.xlabel("Epochs")
        plt.savefig(self.exp_dir+'/'+self.exp_name+"/out_log/StateLoss.png", dpi = 256, facecolor = 'w', bbox_inches='tight')


