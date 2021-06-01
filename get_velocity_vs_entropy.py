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
from util import rolling_mean, probability, get_mPE_matrix
from sklearn.cluster import KMeans

###################################################################################################################################################
################################################################ LOAD TRAJECTORIES ################################################################
###################################################################################################################################################

# Calculate instantaneous velocity of each datapoint (maybe only consider x-y dims)
lest_varaince_dim = 8*3
minutes = 6
fs = 300
bin_length = fs*60*minutes

bins_number = int(reduced_traj.shape[0]/bin_length)
traj_number = 30
orders = [5]
vel_matrix = get_vel_matrix(all_trajectories, bins_number, traj_number)

# plotting bin-velocity against corresponding mPE
print(bins_number)
print(vel_vector.shape)
print(mPE_vector.shape)
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