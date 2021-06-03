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
from util import rolling_mean, probability, get_mPE_matrix, get_vel_matrix
from sklearn.cluster import KMeans


######################## params ########################
apply_PCA = True
n_PC = 
 
###################################################################################################################################################
################################################################ LOAD TRAJECTORIES ################################################################
###################################################################################################################################################

print('########################## LOADING TRAJECTORIES ##########################')

sub_sampling = 50
modes = ['normal', 'drug']
root_dir = '/rds/general/user/lr4617/home/4th_Year_Project/CAPTURE_rat_multidimensional/raw_data/'
# load entire high-dimensional trajectories
cnt = 0
lengths = []
for mode in modes:
    trajs = os.listdir(root_dir + mode + '/' )
    for traj_n in trajs:
        if traj_n != '.ipynb_checkpoints': 
            # loading entire high-dimensional trajectory
            path = root_dir + mode + '/' + traj_n + '/' + 'trajectories_na/'
            trajectories = os.listdir(path)
            # removing NaN columns
            nan_cols = []
            for i, time_bin in enumerate(trajectories):
                if time_bin != 'behavs' and time_bin != '.ipynb_checkpoints':
                    trajectory = loadmat(path + time_bin)
                    trajectory = trajectory['trajectory'] 
                    for i in range(trajectory.shape[1]):
                        if np.isnan(trajectory[:, i]).all():
                            nan_cols.append(i)

            # create entire trajectory
            nan_cols = np.asarray(nan_cols)
            if nan_cols.size > 0:
                if len(np.where(nan_cols==nan_cols[0])[0])*3 == len(nan_cols):
                    all_trajectories =  np.zeros( (int((trajectory.shape[0]*len(trajectories))), trajectory.shape[1]-len(nan_cols)) )
                    sampled_trajectories =  np.zeros( (int((trajectory.shape[0]*len(trajectories))/sub_sampling), trajectory.shape[1]-len(nan_cols)) )
            else:
                all_trajectories =  np.zeros( (int((trajectory.shape[0]*len(trajectories))), trajectory.shape[1]-len(nan_cols)) )
                sampled_trajectories = np.zeros( (int((trajectory.shape[0]*len(trajectories))/sub_sampling), trajectory.shape[1]) )

            window = sub_sampling
            for i, time_bin in enumerate(trajectories):
                if time_bin != 'behavs' and time_bin != '.ipynb_checkpoints':
                    trajectory = loadmat(path + time_bin)
                    trajectory = trajectory['trajectory'] 
                    # idx = np.round(np.arange(0, trajectory.shape[0], sub_sampling)).astype(int)
                    idx_2 = i*trajectory.shape[0]
                    all_trajectories[idx_2:idx_2+trajectory.shape[0], 0:trajectory.shape[1]] = trajectory
                    
            print(all_trajectories.shape)
            
            # convert nan to number when not it is a sparse recurrence (not an entire COLUMN)
            all_trajectories = np.nan_to_num(all_trajectories)
            lengths.append(all_trajectories.shape[0])
                        
            # append trajectory to all trajectories
            if cnt==0:
                rats = all_trajectories
            if cnt>0:
                rats = np.concatenate((rats, all_trajectories), axis=0)
                
            cnt += 1

print(rats.shape)

###################################################################################################################################################
############################################################ Velocity vs Entropy ##################################################################
###################################################################################################################################################


for which_traj, length in enumerate(lengths):
    print('TRAJECTORY ', str(which_traj))
    # inspecting inter-dimensional variance with PCA
    if which_traj == 0:
        idx = 0
    else:
        idx += lengths[which_traj-1]

    traj = rats[idx:idx+length, :]

    # Calculate instantaneous velocity of each datapoint (maybe only consider x-y dims)
    fs = 300
    minutes = 5
    bin_length = fs*60*minutes
    unit_length = 1500
    traj_number = int(bin_length/unit_length)
    bins_number = int(traj.shape[0]/bin_length)
    orders = [3, 4]

    for order in orders:

        mPE_vector, vel_matrix = get_vel_matrix(all_trajectories, bins_number, traj_number, order)

        # plotting bin-velocity against corresponding mPE
        vel_vector = vel_matrix.flatten()
        mPE_vector_ = mPE_vector.flatten()
        p = np.polyfit(mPE_vector_, vel_vector, 2)
        x_new = np.linspace(4,12,200)
        ffit = np.polyval(p, x_new)

        fig = plt.figure()
        plt.scatter(mPE_vector_, vel_vector)
        plt.xlabel('mPE')
        plt.ylabel('velocity')
        plt.plot(x_new, ffit)

        path_out = '/rds/general/user/lr4617/home/4th_Year_Project/Final_Report/entropy_analysis/mPE_vs_velocity/' + mode + '_order_' + str(order) + '_traj_' + str(traj_n) + 'png'
        plt.savefig(path_out)
        plt.show()

