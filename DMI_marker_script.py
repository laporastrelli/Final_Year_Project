#########################################
################ IMPORTS ################
#########################################
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
from util import rolling_mean, probability, probability_v2, get_mPE_matrix
from sklearn.cluster import KMeans



###########################################
################ LOAD DATA ################
###########################################

rats = np.load('/rds/general/user/lr4617/home/4th_Year_Project/Final_Report/control_analysis/rats_sampling_10_window_150.npy')
print(rats.shape)
lengths = np.load('/rds/general/user/lr4617/home/4th_Year_Project/Final_Report/control_analysis/lengths.npy')

to_be_removed = '.ipynb_checkpoints'

root_dir = '/rds/general/user/lr4617/home/4th_Year_Project/CAPTURE_rat_multidimensional/raw_data/'
normal_trajs = os.listdir(root_dir + 'normal')
drug_trajs_1 = os.listdir(root_dir + 'caffeine')
drug_trajs_2 = os.listdir(root_dir + 'amphetamine')

if to_be_removed in normal_trajs:
    idx = normal_trajs.index(to_be_removed)
    normal_trajs.pop(idx)
    
if to_be_removed in drug_trajs_1:
    idx = drug_trajs_1drug_trajs_1.index(to_be_removed)
    drug_trajs_1.pop(idx)

if to_be_removed in drug_trajs_2:
    idx = drug_trajs_2.index(to_be_removed)
    drug_trajs_2.pop(idx)

names = normal_trajs + drug_trajs_1 + drug_trajs_2



#####################################################
################ DMI: TIME-DEPENDENT ################
#####################################################

DMIs = np.zeros((20,100,3))

modes = ['normal', 'caffeine', 'amphetamine']
sample_size = 34000
min_length = 334800
n_PC = 3
path_out = '/rds/general/user/lr4617/home/4th_Year_Project/Final_Report/DMIs_in_time.npy'  

################## CHANGE ##################
dims = np.arange(60)
############################################


for which_traj, length in enumerate(lengths):
    
    ##### retireving single trajectory #####
    if which_traj == 0:
        idx = 0
    else:
        idx += lengths[which_traj-1]

    traj = rats[idx:idx+length, dims]
    
    
    pca = PCA(n_components=n_PC)
    reduced_traj = pca.fit_transform(traj)
    
    reduced_traj = reduced_traj[0:min_length, :]
    traj = traj[0:min_length, :]
    
    if which_traj%4 == 0:
        rats_reduced = np.transpose(reduced_traj)
        rats_ = np.transpose(traj)
        
        print(rats_reduced.shape, rats_.shape)
    else:
        rats_reduced = np.concatenate((rats_reduced, np.transpose(reduced_traj)), axis=1)
        rats_ = np.concatenate((rats_, np.transpose(traj)), axis=1)

    print(rats_reduced.shape, rats_.shape)

    if (which_traj+1)%4 == 0:
        
        print("#################################################################################")

        for iii in range(0, 4*min_length, 4*sample_size):

            red_signal = rats_reduced[:, iii:iii+4*sample_size,]

            for dim in range(int(len(dims)/3)):

                print('dimension = ', dim)

                # retrieve marker signal (x,y,z)
                marker_signal = rats_[dim:dim+3, iii:iii+4*sample_size]

                print(marker_signal.shape, rats_reduced.shape)

                # calculate marker signal mPE
                [mH, _] = mPE_(marker_signal, 3)

                
                # create joint signal for Joint Dynamical Entropy
                joint_signal = np.concatenate((red_signal, marker_signal), axis=0)
                print(joint_signal.shape)

                # caluclate joint entropy using joint signal
                [JH, _] = mPE_(joint_signal, 3)

                # calculate dynamical mutual information
                DMI = JH - mH

                print(DMI, JH, mH)

                # fill array
                DMIs[dim, int(iii/(4*sample_size)), int((which_traj+1)/4)] = DMI

            
np.save(path_out, DMIs)         
