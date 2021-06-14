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


rats = np.load('/rds/general/user/lr4617/home/4th_Year_Project/Final_Report/control_analysis/rats_sampling_10_window_150_ordered.npy')
lengths = np.load('/rds/general/user/lr4617/home/4th_Year_Project/Final_Report/control_analysis/lengths.npy')

to_be_removed = '.ipynb_checkpoints'

root_dir = '/rds/general/user/lr4617/home/4th_Year_Project/CAPTURE_rat_multidimensional/raw_data/'
normal_trajs = os.listdir(root_dir + 'normal')
drug_trajs_1 = os.listdir(root_dir + 'caffeine')
drug_trajs_2 = os.listdir(root_dir + 'amphetamine')


if to_be_removed in normal_trajs:
    idx = normal_trajs.index(to_be_removed)
    normal_trajs.pop(idx)
    normal_trajs.sort()

    
if to_be_removed in drug_trajs_1:
    idx = drug_trajs_1.index(to_be_removed)
    drug_trajs_1.pop(idx)
    drug_trajs_1.sort()

if to_be_removed in drug_trajs_2:
    idx = drug_trajs_2.index(to_be_removed)
    drug_trajs_2.pop(idx)
    drug_trajs_2.sort()

names = normal_trajs + drug_trajs_1 + drug_trajs_2

DMIs = np.zeros((len(lengths), 20, 3))
n_PC = 3
max_length = 348000


for which_traj, length in enumerate(lengths):

    print('TRAJECTORY ', str(which_traj))
    
    # retrieve trajectory from trajectories
    if which_traj == 0:
        idx = 0
    else:
        idx += lengths[which_traj-1]

    traj = rats[idx:idx+length, :]

    print('########################## APPLY PCA ##########################')
    
    # apply PCA to high-d signal to reduce it to "n_PC" dims
    pca = PCA(n_components=n_PC)
    reduced_traj = pca.fit_transform(traj)

    # formatting trajectories
    reduced_traj = reduced_traj[0:max_length, :]
    
    # calculate marker signal mPE
    [HH, _]=mPE_(reduced_traj, 3)
    
    for dim in range(0,traj.shape[1],3):
        
        # retrieve marker signal (x,y,z) and formatting to coherent length
        marker_signal = traj[0:max_length, dim:dim+3]
        
        # calculate marker signal mPE
        [mH, _] = mPE_(marker_signal, 3)
        
        # create joint signal for Joint Dynamical Entropy
        joint_signal = np.concatenate((reduced_traj, marker_signal), axis=1)
        
        # caluclate joint entropy using joint signal
        [JH, _] = mPE_(joint_signal, 3)
        
        # calculate dynamical mutual information
        DMI = HH + mH - JH

        inf = np.array([JH, mH, HH])
        
        DMIs[which_traj, int(dim/3), : ] = inf
        
        print(DMI[0])

path_out = '/rds/general/user/lr4617/home/4th_Year_Project/Final_Report/'
name_out = 'marker_DMI_' + str(n_PC) + 'PC.npy'
np.save(path_out + name_out, DMIs)