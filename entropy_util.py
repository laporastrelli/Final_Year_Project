import numpy as np
import math 

def conditional_entropy(prob_s1s2, prob_s2):
    E_cond = 0
    for i in range(prob_s1s2.shape[0]):
        for j in range(prob_s1s2.shape[1]):
            if prob_s1s2[i,j] > 0:
                E_cond += prob_s1s2[i,j] * math.log((prob_s2[j]/prob_s1s2[i,j]), 2)
            
    return E_cond

def joint_entropy(prob_s1s2):
    E_joint = 0
    for i in range(prob_s1s2.shape[0]):
        for j in range(prob_s1s2.shape[1]):
            if prob_s1s2[i,j] > 0:
                E_joint += prob_s1s2[i,j] * math.log((1/prob_s1s2[i,j]), 2)
            
    return E_joint