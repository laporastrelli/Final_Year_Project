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

sns.set()
sns.set_style("white")
sns.set_style("ticks")

# params
orders = [3, 4]
n_PCs = [2, 3, 4, 6]
bound_len = 5e06
n_trials = 20
length_ = 10

cnt = 0
fig, axs = plt.subplots(len(orders), len(n_PCs), figsize=(14, 6))

labels = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)']
sizes = [100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 500000]

to_save = np.zeros((n_trials, len(sizes), len(orders)*len(n_PCs)))

# convergence analysis over 20 trials
for i, order in enumerate(orders):
    
    for j, n_PC in enumerate(n_PCs):
        
        if n_PC >=4:
            sizes = [100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 500000]
        else:
            sizes = [100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 500000]
    
        H_bound = np.log2(mt.factorial(order)**n_PC)

        # initialize sample entropy array
        sample_H = np.zeros((n_trials, len(sizes)))

        for trial in range(n_trials):
            f_unit = np.random.rand(n_PC, length_)

            for jj, size in enumerate(sizes):
                sample_traj = np.zeros((n_PC, size))

                for iii in range(int(size/length_)):

                    # create trajectory for entropy array
                    idx = iii * length_
                    sample_traj[:, idx:idx+length_] = f_unit

                    # update fundamental unit 
                    f_unit = np.random.rand(n_PC, length_)

                #######################################################################
                [H_sample, _] = mPE_(sample_traj, order)
                sample_H[trial, jj] = H_sample
                #######################################################################

        
        axs[i, j].plot(sizes, np.mean(sample_H, axis=0))
        axs[i, j].errorbar(sizes, np.mean(sample_H, axis=0), yerr=np.var(sample_H, axis=0), fmt="o", color="r")
        axs[i, j].axhline(y=H_bound, color="black", linestyle="--")

        if j == 0:
            axs[i, j].set_ylabel('Order ' + str(order) + ' mPE')
        else:
            axs[i, j].set_ylabel('')

        axs[i, j].set_xscale("log")
        axs[i, j].set_xlim([50, 1e06])
        
        x_min, x_max, y_min, y_max = axs[i, j].axis()
        if y_max - y_min < 0.3:
            axs[i][j].set_ylim([H_bound - 0.3,  + H_bound + 0.1])
    
        axs[i, j].grid()
        axs[i, j].text(-0.05, 1.13, labels[cnt], transform=axs[i, j].transAxes, fontsize=16, fontweight='bold', va='top', ha='right') 
        
        to_save[:, :, cnt] = sample_H

        cnt += 1

        print(cnt)

name_out_ = 'bias_reduction_data.npy'
path_out_ = '/rds/general/user/lr4617/home/4th_Year_Project/Final_Report/control_analysis/'
np.save(path_out_+name_out_, to_save)

fig.text(0.5, 0.02, 'Sample Size', ha='center')
name_out = 'bias_reduction'
path_out = '/rds/general/user/lr4617/home/4th_Year_Project/Final_Report/control_analysis/'
plt.savefig(path_out +  name_out  + ".png")

