import numpy as np
import csv, h5py, json, pickle
import torch
from torch.utils.data import DataLoader
from src_param.PreProc_Data.DataProc import StackedSequenceDataset


class DynSystem_Data:

    def load_and_preproc_data(self):
        '''
        loads and preprocesses data
        Requires
        --------
        data_dir, norm_input
        Generates
        ---------
        lp_data (numpy tensor): [num_traj, timesteps, statedim] Loaded Data
        data_args (dict)      :  Attributes of the loaded data
        '''
        
        self.lp_data   = np.load(self.data_dir)

        #For KS
        if self.dynsys == "KS": 
            self.lp_data = self.lp_data[:,::self.time_sample,:]
            self.lp_data = self.lp_data[:,self.ntransients:,:]
        
        #For 2D Cylinder Flow
        elif self.dynsys == "2DCyl":
            self.lp_data = self.lp_data[:,self.ntransients:self.nenddata,:]

        #additional data parameters
        self.statedim   = self.lp_data.shape[2:]
        self.state_ndim = len(self.statedim)
        self.statedim   = self.statedim[0] - self.param_dim

        print("State Dims: ", self.statedim)

        #Normalising Data
        if self.norm_input:
            print("normalizing Input along parameter dimension")
            mean = self.lp_data.mean(axis=(1, 2), keepdims=True)  # Shape: [parameters, 1, 1]
            std = self.lp_data.std(axis=(1, 2), keepdims=True) # Shape: [parameters, 1, 1]

            self.lp_data = (self.lp_data - mean)/std
  
    
    def create_dataset(self, mode = "Both"):

        '''
        Creates non sequence dataset for state variables and divides into test, train and val dataset
        Requires
        --------
        lp_data: [num_traj, timesteps, statedim] state variables
        mode   : "Train" for only train dataset, "Test" for only test dataset, "Both" for both datset

        Returns
        -------
        Dataset : [num_traj, timesteps, statedim] Input , Output (both test and train)
        Dataloader: [num_traj*timesteps, statedim] 
        '''

        if mode == "Both" or mode == "Train":
            
            if self.dynsys == "KS":
                self.train_data = self.lp_data[:,:int(self.train_size * self.lp_data.shape[1])]
            elif self.dynsys == "2DCyl":
                self.train_data = self.lp_data[:,::2]


            self.train_num_trajs = self.train_data.shape[0]
            print("Train_Shape: ", self.train_data.shape)
            self.train_dataset    = StackedSequenceDataset(self.train_data, self.__dict__)
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle = True, num_workers = 0)
        
        # print("out of train")
        if mode == "Both" or mode == "Test":
            
            if self.dynsys == "KS":
                self.test_data  = self.lp_data[:,int(self.train_size * self.lp_data.shape[1]):]
            
            elif self.dynsys == "2DCyl":
                self.test_data = self.lp_data[:,1::2]
         
            
            print("Test_Shape: " , self.test_data.shape)
            self.test_num_trajs  = self.test_data.shape[0]
            self.test_dataset     = StackedSequenceDataset(self.test_data , self.__dict__)
            self.test_dataloader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle = False, num_workers = 0)
