import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io
import scipy.special as psi

from scipy.io import loadmat
from scipy import stats
from scipy.spatial import distance
from scipy.stats import entropy
from prob_util import *
from entropy_util import *

modes = ['normal', 'drug', 'vehicle']
root_dir = '/rds/general/user/lr4617/home/4th_Year_Project/CAPTURE_rat_multidimensional/raw_data/'
for mode in modes:
    trajs = os.listdir(root_dir + mode + '/')
    for traj_n in trajs:
        # loading entire high-dimensional trajectory
        path = root_dir + mode + '/' + traj_n + '/'
        path = '/rds/general/user/lr4617/home/4th_Year_Project/CAPTURE_rat_multidimensional/raw_data/normal/traj_1/trajectories_na/'
        trajectories = os.listdir(path)

    # removing invalid values (e.g. NaN)
    # input data is already normalized (z-score) but needs to get rid of non-valued datapoints
    nan_cols = []
    for i, time_bin in enumerate(trajectories):
        if time_bin != 'behavs':
            trajectory = loadmat(path + time_bin)
            trajectory = trajectory['trajectory'] 
            for i in range(trajectory.shape[1]):
                if np.isnan(trajectory[:, i]).all():
                    nan_cols.append(i)

    print(nan_cols)

    nan_cols = np.asarray(nan_cols)
    if len(nan_cols)>0:
        if len(np.where(nan_cols==nan_cols[0])[0])*3 == len(nan_cols):
            all_trajectories = np.zeros((trajectory.shape[0]*int(len(trajectories)), trajectory.shape[1]-3))
            for i, time_bin in enumerate(trajectories):
                if time_bin != 'behavs':
                    trajectory = loadmat(path + time_bin)
                    trajectory = trajectory['trajectory'] 
                    trajectory = np.delete(trajectory, nan_cols, 1)
                    idx_2 = i*trajectory.shape[0]
                    all_trajectories[idx_2:idx_2+trajectory.shape[0], 0:trajectory.shape[1]] = trajectory
    else:
        all_trajectories = np.zeros((trajectory.shape[0]*int(len(trajectories)), trajectory.shape[1]))
        for i, time_bin in enumerate(trajectories):
            if time_bin != 'behavs':
                trajectory = loadmat(path + time_bin)
                trajectory = trajectory['trajectory'] 
                idx_2 = i*trajectory.shape[0]
                all_trajectories[idx_2:idx_2+trajectory.shape[0], 0:trajectory.shape[1]] = trajectory

    print(all_trajectories.shape)

    fs_og = 300
    length_ = int(all_trajectories.shape[0]/len(trajectories))
    shifts = np.arange(0, 300, 5)
    decimals = 2
    dim = 6

    auto_MI_per_lag = np.zeros((len(shifts), 1))
    auto_MI_per_lag_norm = np.zeros((len(shifts), 1))
    path_out = '/rds/general/user/lr4617/home/4th_Year_Project/CAPTURE_rat_multidimensional/raw_data/normal/traj_1/'

    s1 = all_trajectories[length_:2*length_, dim]
    prob_s1 = probability(s1, decimals)

    for i, shift in enumerate(shifts):
        print("SHIFT = ", shift)

        s2 = all_trajectories[length_+shift:2*length_+shift, dim]
        prob_s2 = probability(s2, decimals)
        prob_s1s2 = joint_probability(s1, s2, decimals) 
        auto_MI = AIMF(prob_s1, prob_s2, prob_s1s2)
        print('auto mutual information: ', auto_MI)
        auto_MI_per_lag[i] = auto_MI
        np.save(path_out + 'na_auto_MI_per_lag_' + str(dim), auto_MI_per_lag)

    auto_MI_per_lag_norm = auto_MI_per_lag/(np.max(auto_MI_per_lag))

    # plotting relation between MI and sub-sampling frequency
    fig = plt.figure()
    plt.scatter(shifts, auto_MI_per_lag)
    plt.ylabel('Auto-MI')
    plt.xlabel('Time Lag')
    plt.title('AUTO MI vs TIME LAG')
    plt.savefig(path_out + "na_auto_MI_per_lag_" + str(dim) + ".png")
