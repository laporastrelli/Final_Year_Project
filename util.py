import numpy as np
from mPE_fn import mPE, mPE_
from scipy.spatial import distance


def probability(*argv):
    '''
    input: 
        - 1D sequence of rv observations
    return: 
        - probability vector
    '''
    n_args = len(argv)
    if n_args == 1:
        sequence = argv[0]
        decimals = 1
        size = 0
        
    if n_args == 2:
        sequence = argv[0]
        decimals = argv[1]
        size = 0
    
    if n_args == 3:
        sequence = argv[0]
        decimals = argv[1]
        size = argv[2]
    
    if len(sequence.shape) > 1 and (sequence.shape[0] < sequence.shape[1]):
        sequence = np.transpose(sequence)
    
    # round input sequence to avoid sparse probability vector
    sequence = np.round(sequence, decimals)
    unique = np.unique(sequence, axis=0)
    n_unique = len(unique)

    # fill probability vector
    if size == 0:
        prob_vector = np.zeros((n_unique, 1))
    else:
        prob_vector = np.zeros((size, 1))

    for row in sequence:
        if len(sequence.shape) > 1:
            occurrences = len(np.where(np.all(np.isclose(sequence, row), axis=1))[0])
            idx = np.where(np.all(np.isclose(unique, row), axis=1))[0][0]
        else:
            occurrences = len(np.where(np.isclose(sequence, row))[0])
            idx = np.where(np.isclose(unique, row))[0][0]
            
        if prob_vector[idx] == 0:
            prob_vector[idx] = occurrences/(sequence.shape[0])
            
    return prob_vector


def probability_v2(sequence, centroids):
    '''
    input: 
        - 1D sequence of rv observations
        - centroids (based on k-means clustering)
    return: 
        - probability vector
    '''
    sequence = np.asarray(sequence)
    
    if len(sequence.shape) > 1 and (sequence.shape[0] > sequence.shape[1]):
        sequence = np.transpose(sequence)
        
    if centroids.shape[0] > centroids.shape[1]:
        centroids = centroids.transpose()
    
    n_unique = centroids.shape[1]
    prob_vector = np.zeros((1, n_unique))

    for elem in sequence:
        elem_vector = elem*np.ones((n_unique, ))
        diff = np.absolute(elem_vector - centroids)
        idx = np.where(diff[0]==np.min(diff))[0]
        prob_vector[:, idx] += 1
        
    prob_vector = prob_vector/(sequence.shape[0])
            
    return prob_vector[0]


def get_mPE_matrix(reduced_traj, bins_number, traj_number, orders, random):
    if random:
        bins_number = 2
        mPE_vector = np.zeros((bins_number, traj_number, len(orders)))
        traj_length = int((reduced_traj.shape[0]/bins_number)/traj_number)
        print(mPE_vector.shape)

        for i in range(bins_number):
            idx = 0
            for j in range(0, traj_length*traj_number, traj_length):
                print(idx)
                idx_1 = np.random.randint(np.max(reduced_traj.shape) - traj_length)
                traj = reduced_traj[idx_1: idx_1 + traj_length]
                [HH, _]=mPE_(traj, orders[0])
                mPE_vector[i, idx, 0] = HH
                idx +=1

    else:
        mPE_vector = np.zeros((bins_number, traj_number, len(orders)))
        traj_length = int((reduced_traj.shape[0]/bins_number)/traj_number)
        
        for a, order in enumerate(orders):
            
            for i in range(bins_number):
                idx = 0
                
                for j in range(0, traj_length*traj_number, traj_length):
                    idx_1 = i*traj_number*traj_length 
                    traj = reduced_traj[idx_1 + j: idx_1 + j + traj_length]
                        
                    if traj.shape[0]>0:
                        [HH, _]=mPE_(traj, order)
                        mPE_vector[i, idx, a] = HH
                        
                    idx +=1
    
    

                
    return mPE_vector



def get_vel_matrix(trajectory, bins_number, traj_number, orders, least_varaince_zdim=0):

    if trajectory.shape[0] < trajectory.shape[1]:
        trajectory =  trajectory.transpose()

    dims = 2

    mPE_vector = np.zeros((bins_number, traj_number, len(orders)))
    vel_matrix = np.zeros((bins_number, traj_number, len(orders)))
    print(bins_number, traj_number)
    traj_length = int((trajectory.shape[0]/bins_number)/traj_number)

    for a, order in enumerate(orders):
        for i in range(bins_number):

            idx = 0
            idx_1 = 0

            for j in range(0, traj_length*traj_number, traj_length):

                idx_1 = i*traj_number*traj_length 
                traj = trajectory[idx_1 + j: idx_1 + j + traj_length, least_varaince_zdim:least_varaince_zdim+dims]
                traj = np.asarray(traj)
                vel_bin = 0
                last_point = traj[0, :]

                for point in traj:
                    vel_bin = vel_bin + distance.euclidean(point, last_point)
                    last_point = point

                if traj.shape[0]>0:
                    [HH, _]=mPE_(traj, order)
                    mPE_vector[i, idx] = HH
                vel_matrix[i, idx, a] = vel_bin/(traj_length/50)

                idx += 1
                
    return mPE_vector, vel_matrix




def rolling_mean(x, window, overlapping=True):
    '''
    input:
        x      - input sequence
        window - rolling window
        
    returns:
        y      - moving averaged sequence
    '''
        
    ## assuming that there are more observations than variables:
    if np.max(x.shape)> x.shape[0]:
        x=x.transpose()
        
    if overlapping:
        y = np.zeros((x.shape))
        for i in range(0, x.shape[0] - window):
            y[i, :] = np.mean(x[i:i+window, :], axis=0)

        y[x.shape[0]-window:x.shape[0]-1, :] = x[x.shape[0]-window:x.shape[0]-1, :]
        
    else:
        y = np.zeros((int(x.shape[0]/window), x.shape[1]))
        windows = np.arange(0, x.shape[0] - window, window)
        for i, t in enumerate(windows):
            y[i, :] = np.mean(x[t:t+window, :], axis=0)
    
    return y