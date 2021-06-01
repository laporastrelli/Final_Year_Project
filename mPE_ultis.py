from scipy.io import savemat
from tempfile import TemporaryFile
import numpy as np
import itertools as it
from scipy.integrate import quad
import math as mt
import scipy.special as psi
import matplotlib.pyplot as plt
plt.style.use('classic')
import seaborn as sns
import pandas as pd
from scipy.io import loadmat
from scipy import stats
from numpy.random import seed
from numpy.random import rand
import matplotlib.gridspec as gridspec


def integrand(t, ni):
     return (t**(ni-1))/(1+t)

# Computes all the permutations of a range 0-k
def permutation(k):
  f_k=mt.factorial(k)
  A=np.empty((f_k,k))
  for i, perm in enumerate(it.permutations(range(k))):
      A[i,:] = perm
  return A

# Converts an array into a list
def array_list(array_num): 
    num_list = array_num.tolist()
    return num_list

# Creates an array that displays which permutation represent the order
# of the input array
def ubble(v):
    n=len(v)
    a=range(0,n)
    B=np.array([v,a])
    t=np.array([0., 0.])
    for i in range(0,n-1):
        for j in range(1,n):
            if B[0,j-1]>B[0,j]:
                
                t[:]=B[:,j]
                B[:,j]=B[:,j-1]
                B[:,j-1]=t[:]    
    ord=B[1][:]
    return ord