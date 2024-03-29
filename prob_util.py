import numpy as np


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
        
    if n_args == 2:
        sequence = argv[0]
        decimals = argv[1]
    
    if len(sequence.shape) > 1 and (sequence.shape[0] < sequence.shape[1]):
        sequence = np.transpose(sequence)
    
    # round input sequence to avoid sparse probability vector
    sequence = np.round(sequence, decimals)
    unique = np.unique(sequence, axis=0)
    n_unique = len(unique)

    # fill probability vector
    prob_vector = np.zeros((n_unique, 1))
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


def joint_probability(*argv):
    '''
    input: 
        - sequence_1 of rv_1 observations
        - sequence_2 of rv_2 observations
        - number of dimensions of input sequences to consider(default = all)
    return: 
        - joint probability vector
    '''
    n_args = len(argv)
    if n_args == 2:
        s1 = argv[0]
        s2 = argv[1]
        if len(s1.shape)==1 and len(s2.shape)==1:
            s1 = np.reshape(s1, (s1.shape[0], 1))
            s2 = np.reshape(s2, (s2.shape[0], 1))
        dims = s1.shape[1]
        decimals = 1
    
    if n_args == 3:
        s1 = argv[0]
        s2 = argv[1]
        decimals = argv[2]
        if len(s1.shape)==1 and len(s2.shape)==1:
            s1 = np.reshape(s1, (s1.shape[0], 1))
            s2 = np.reshape(s2, (s2.shape[0], 1))
        dims = s1.shape[1]
        
    if n_args == 4:
        s1 = argv[0]
        s2 = argv[1]
        decimals = argv[2]
        dims = argv[3]
        # select dims based on input
        s1 = s1[:, 0:dims]
        s2 = s2[:, 0:dims]
        
    # checking that the dimensions of the input sequences are in the right order
    if s1.shape[0] < s1.shape[1]:
        s1 = np.transpose(s1)
    if s2.shape[0] < s2.shape[1]:
        s2 = np.transpose(s2)
    
    s1 = np.around(s1, decimals)
    s2 = np.around(s2, decimals) 
    
    # here we assume that the input sequences are already rounded (n_observations x n_dimensions)
    unique_s1 = np.unique(s1, axis=0)
    n_triplets_s1 = len(unique_s1)
    unique_s2 = np.unique(s2, axis=0)
    n_triplets_s2 = len(unique_s2)

    joint_data = np.concatenate((s1, s2), axis=1)
    
    # filling joint probability matrix
    joint_prob_matrix = np.zeros((n_triplets_s1, n_triplets_s2))
    occurrences, idx_s1, idx_s2 = 0, 0, 0
    for joint_array in joint_data:
        occurrences = len(np.where(np.all(np.isclose(joint_data, joint_array), axis=1))[0])
        idx_s1 = np.where(np.all(np.isclose(unique_s1, joint_array[0:dims]), axis=1))
        idx_s2 = np.where(np.all(np.isclose(unique_s2, joint_array[dims:2*dims]), axis=1))
        if joint_prob_matrix[idx_s1[0][0], idx_s2[0][0]] == 0:
            joint_prob_matrix[idx_s1[0][0], idx_s2[0][0]] = (occurrences/joint_data.shape[0])
            
    return joint_prob_matrix





