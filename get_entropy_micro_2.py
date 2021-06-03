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
from numpy.random import normal, seed
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


###################################################################################################################################################
#################################################################### PARAMS #######################################################################
###################################################################################################################################################
n_PC = 3
significance_level = 0.01
decimals = 3
use_all = True
n_clusters_kmeans = 50
unit_length_entropy = 1500

###################################################################################################################################################
################################################################ LOAD TRAJECTORIES ################################################################
###################################################################################################################################################

print('########################## LOADING TRAJECTORIES ##########################')

modes = ['normal', 'drug']
names = []
root_dir = '/rds/general/user/lr4617/home/4th_Year_Project/CAPTURE_rat_multidimensional/raw_data/'
# load entire high-dimensional trajectories
cnt = 0
lengths = []
for mode in modes:
    trajs = os.listdir(root_dir + mode + '/' )
    for traj_n in trajs:
        names.append(traj_n)
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

            # decide whether to use sub-sampling or not
            nan_cols = np.asarray(nan_cols)
            a = 0
            if use_all:
                sub_sampling = 1
                window = 50
            elif not use_all:
                sub_sampling = 50
                window = sub_sampling

            # create trajectory
            if nan_cols.size > 0:
                if len(np.where(nan_cols==nan_cols[0])[0])*3 == len(nan_cols):
                    sampled_trajectories = np.zeros( (int((trajectory.shape[0]*len(trajectories))/sub_sampling), trajectory.shape[1]-len(nan_cols)) )
            else:
                sampled_trajectories = np.zeros( (int((trajectory.shape[0]*len(trajectories))/sub_sampling), trajectory.shape[1]) )

            for i, time_bin in enumerate(trajectories):
                if time_bin != 'behavs':
                    trajectory = loadmat(path + time_bin)
                    trajectory = trajectory['trajectory'] 
                    if nan_cols.size > 0:
                        trajectory = np.delete(trajectory, nan_cols, 1)
                    
                    mov_av_traj = rolling_mean(trajectory, window, overlapping=True)
                    idx_2 = i*mov_av_traj.shape[0]
                    sampled_trajectories[idx_2:idx_2+mov_av_traj.shape[0], 0:sampled_trajectories.shape[1]] = mov_av_traj
                    
            # convert nan to number when not it is a sparse recurrence (not an entire column)
            sampled_trajectories = np.nan_to_num(sampled_trajectories)
            lengths.append(sampled_trajectories.shape[0])

            # append trajectory to all trajectories
            if cnt==0:
                rats = sampled_trajectories
            if cnt>0:
                rats = np.concatenate((rats, sampled_trajectories), axis=0)
                
            cnt += 1

            print(rats.shape)


###################################################################################################################################################
####################################################################### PCA #######################################################################
###################################################################################################################################################
count = 0
for which_traj, length in enumerate(lengths):

    if count <= 3:
        mode = 'normal'
    else:
        mode = 'drug'

    name = names[count]

    measures = np.zeros((7, 50))    
    # consists of:
    # - explained variance
    # - p-value (control)
    # - j-s (control)
    # - p-value(result)
    # - j-s (result)
    # - mean mPE
    # - std mPE
    
    print('TRAJECTORY ', str(which_traj))
    # inspecting inter-dimensional variance with PCA
    if which_traj == 0:
        idx = 0
    else:
        idx += lengths[which_traj-1]

    traj = rats[idx:idx+length, :]

    print('########################## APPLY PCA ##########################')

    pca = PCA()
    pca.fit(traj)

    explained_var_plot = [np.sum(pca.explained_variance_ratio_[0:i+1]) for i in range(len(pca.explained_variance_ratio_))]
    measures[0, 0:len(explained_var_plot)] = pca.explained_variance_ratio_

    plt.scatter(np.arange(traj.shape[1]), explained_var_plot)
    plt.ylabel('Cumulative Expalined Variance')
    plt.xlabel('Dimension')

    ##### IMAGES ARE SAVED IN TRAJECTORY FOLDER #####
    path_out = '/rds/general/user/lr4617/home/4th_Year_Project/Final_Report/entropy_analysis/mPE_vs_time/micro/' + str(n_PC) + 'PC/control/' + mode + '/' + name + '/'
    name_out = mode + '_explained_var_' + name
    plt.savefig(path_out +  name_out  + ".png")
    plt.show()

    # reduce data according to explained variance values using linear PCA
    pca = PCA(n_components=n_PC)
    reduced_traj = pca.fit_transform(traj)




    ###################################################################################################################################################
    ########################################################### CONTROL ANALYSIS ######################################################################
    ###################################################################################################################################################

    print('########################## CONTROL ANALYSIS ##########################')

    unit_length = unit_length_entropy
    minutes = 5
    bin_length = 300*60*minutes
    bins_number = int(reduced_traj.shape[0]/bin_length)
    traj_number = int(bin_length/unit_length)
    orders = [3]
    random = True

    too_low = True
    while too_low == True:

        ########## Calculating mPE ##########
        mPE_vector = get_mPE_matrix(reduced_traj, bins_number, traj_number, orders, random)

        ########## plotting mPE continuous distribution as a function of time ##########
        fig = plt.figure()
        c = ['r', 'b']

        ax = sns.kdeplot(mPE_vector[0, :, 0], color=c[0])
        ax = sns.kdeplot(mPE_vector[1, :, 0], color=c[1])

        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Order [3] mPE (bits)')
        plt.ylabel('Estimated Density Function')

        path_out = '/rds/general/user/lr4617/home/4th_Year_Project/Final_Report/entropy_analysis/mPE_vs_time/micro/' + str(n_PC) + 'PC/control/' + + mode + '/' + name + '/'
        name_out = mode + '_control_' + name

        plt.savefig(path_out +  name_out  + ".png")
        plt.show()

        ########## calculate p-value ##########
        [_, p_value] = stats.ks_2samp(mPE_vector[0, :, 0], mPE_vector[1, :, 0])

        if p_value > 0.8:
            too_low = False
            break
    
    path_out = '/rds/general/user/lr4617/home/4th_Year_Project/Final_Report/entropy_analysis/mPE_vs_time/micro/' + str(n_PC) + 'PC/' + mode + '/mPE_control_' + name + '.npy'
    np.save(path_out, mPE_vector)

    ########## Retrieve maxiumum number of elements based on a raw decimal discretization ##########
    data = mPE_vector.flatten()
    kmeans = KMeans(n_clusters=n_clusters_kmeans).fit(data.reshape(-1,1))
    kmeans.predict(data.reshape(-1,1))
    centroids = kmeans.cluster_centers_
    centroids = [centroids[i] for i in range(len(centroids))]
    centroids = np.asarray(centroids)

    ########## Get probability vectors ##########
    prob1 = probability_v2(mPE_vector[0, :, 0], centroids)
    prob2 = probability_v2(mPE_vector[1, :, 0], centroids)

    ########## Evaluation ##########
    js_distance = distance.jensenshannon(prob1, prob2)

    measures[1, 0] = p_value
    measures[2, 0] = js_distance[0]




    ###################################################################################################################################################
    ########################################################### ENTROPY ANALYSIS ######################################################################
    ###################################################################################################################################################


    print('########################## ENTROPY ANALYSIS ##########################')

    ########## Entropy Calculation ##########
    unit_length = unit_length_entropy
    minutes = 5
    bin_length = 300*60*minutes
    bins_number = int(reduced_traj.shape[0]/bin_length)
    traj_number = int(bin_length/unit_length)
    orders = [3]

    ########## Calculate mPE vector ##########
    mPE_vector = get_mPE_matrix(reduced_traj, bins_number, traj_number, orders, random=False)
    path_out = '/rds/general/user/lr4617/home/4th_Year_Project/Final_Report/entropy_analysis/mPE_vs_time/micro/' + str(n_PC) + 'PC/' + mode + '/mPE_result_' + name + '.npy'
    np.save(path_out, mPE_vector)

    ########## Retrieve maxiumum number of elements based on k-means discretization ##########
    data = mPE_vector.flatten()
    kmeans = KMeans(n_clusters=n_clusters_kmeans).fit(data.reshape(-1,1))
    kmeans.predict(data.reshape(-1,1))
    centroids = kmeans.cluster_centers_
    centroids = [centroids[i] for i in range(len(centroids))]
    centroids = np.asarray(centroids)

    ########## plotting mPE continuous distribution as a function of time ##########        
    fig = plt.figure()
    compare_to = mPE_vector[0, :, 0]
    p_value_array = np.zeros((bins_number-1, ))
    js_array = np.zeros((bins_number-1, ))
    means = np.zeros((bins_number, ))
    means[0] = np.mean(compare_to)
    stds = np.zeros((bins_number, ))
    stds[0] = np.std(compare_to)

    for bin_n in range(1, bins_number):
        [_, p_value] = stats.ks_2samp(mPE_vector[bin_n, :, 0], compare_to)
        p_value_array[bin_n -1] = p_value

        prob1 = probability_v2(compare_to, centroids)
        prob2 = probability_v2(mPE_vector[bin_n, :, 0], centroids)

        js_distance = distance.jensenshannon(prob1, prob2)
        js_array[bin_n -1] = js_distance[0]   

        if p_value <= significance_level:
            ax = sns.kdeplot(mPE_vector[bin_n, :, 0])
            compare_to = mPE_vector[bin_n, :, 0]
            
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Order [3] mPE (bits)')
    plt.ylabel('Estimated Density Function')

    path_out = '/rds/general/user/lr4617/home/4th_Year_Project/Final_Report/entropy_analysis/mPE_vs_time/micro/' + str(n_PC) + 'PC/result/' + mode + '/' + name + '/'
    name_out = mode + '_result_' + name

    plt.savefig(path_out +  name_out  + ".png")    
    plt.show()

    measures[3, 0: p_value_array.shape[0]] = p_value_array
    measures[4, 0: js_array.shape[0]] = js_array

    measures[5, 0:means.shape[0]] = means
    measures[6, 0:stds.shape[0]] = stds

    path_out = '/rds/general/user/lr4617/home/4th_Year_Project/Final_Report/entropy_analysis/mPE_vs_time/micro/' + str(n_PC) + 'PC/' + mode + '/specs_' + name + '.npy'
    np.save(path_out, measures)

    count +=1

    

    

    