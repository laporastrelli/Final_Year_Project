import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io
import itertools as it
import scipy.special as psi
plt.style.use('classic')
import seaborn as sns
import pandas as pd
import math as mt
import time
import sys

sys.path.insert(1, '/rds/general/user/lr4617/home/4th_Year_Project/Final_Year_Project/')

from scipy.io import loadmat
from scipy import stats
from numpy.random import seed
from numpy.random import rand
from scipy.integrate import quad
from scipy.io import savemat
from tempfile import TemporaryFile
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.decomposition import KernelPCA
from mpl_toolkits import mplot3d
from mPE_fn import mPE_
from scipy.spatial import distance
from scipy.stats import entropy
from mPE_ultis import integrand, ubble, array_list, permutation

# params
orders = [3, 4]
modes = [1, 2, 3]
lengths = [6, 8, 10]
sizes = [100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 500000]
bound_len = 1e06
n_trials = 20

# convergence analysis over 20 trials, 3 dims and different lengths
for n_PC in range(1,4):
    for i, order in enumerate(orders):
        for length_ in lengths:
    
            # create fundamental unit for synthetic data
            f_unit = np.random.rand(n_PC, length_)
            
            # calculating entropy bound using the entire trajectory length
            rand_traj = np.zeros((n_PC, int(bound_len)))
            
            for ii in range(int(bound_len/length_)):
                
                # create trajectory for entropy bound
                idx = ii * length_
                rand_traj[:, idx:idx+length_] = f_unit
                            
                # update fundamental unit 
                f_unit = np.random.rand(n_PC, length_)                
                            
            #######################################################################
            print("CALCULATING ENTROPY BOUND")
            [H_bound, _] = mPE_(rand_traj, order)
            print("finished")
            #######################################################################
            
            # initialize sample entropy array
            sample_H = np.zeros((n_trials, len(sizes)))
    
            for trial in range(n_trials):
                
                f_unit = np.random.rand(n_PC, length_)

                for j, size in enumerate(sizes):
                    sample_traj = np.zeros((n_PC, size))

                    for iii in range(int(size/length_)):
                                
                        # create trajectory for entropy bound
                        idx = iii * length_
                        sample_traj[:, idx:idx+length_] = f_unit
                                
                        # update fundamental unit 
                        f_unit = np.random.rand(n_PC, length_)

                    #######################################################################
                    print("CALCULATING SAMPLE ENTROPY")
                    [H_sample, _] = mPE_(sample_traj, order)
                    sample_H[trial,j] = H_sample
                    #######################################################################
                    
            path_out = '/rds/general/user/lr4617/home/4th_Year_Project/Final_Report/control_analysis/'
            name_out = str(n_PC) + 'PC_' + str(order) + '_order_' + str(length_) + '_' + 'length'
            plt.figure()
            plt.plot(sizes, np.mean(sample_H, axis=0))
            plt.errorbar(sizes, np.mean(sample_H, axis=0), yerr=np.var(sample_H, axis=0), fmt="o", color="r")
            plt.axhline(y=H_bound, color="black", linestyle="--")
            plt.ylabel('order ' + str(order) + ' mPE')
            plt.xlabel('sample size')
            plt.xscale("log")
            plt.xlim([50, 1e06])
            x_min, x_max, y_min, y_max = plt.axis()
            if y_max - y_min < 0.3:
                plt.ylim([H_bound - 0.3,  + H_bound + 0.1])
            plt.title(str(n_PC) + ' Principal Components')
            plt.grid()
            plt.savefig(path_out +  name_out  + ".png")