from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import torch
import torch.nn.utils.parametrize as parametrize

import cvxpy as cp




def object_func_2(K,C,L,lamb,patient_node_num,num_pat):
    sum = 0
    iter = 0
    for i in range(num_pat):
        c = C[i] #each patient correlation matrix
        num_nodes = patient_node_num[i]
        k = K[iter:iter+num_nodes,iter:iter+num_nodes] #all columns of rows and colums iter+num of nodes + 1 (this gets just that patients correlation matrix in K)
        sum = sum + (np.linalg.norm((k-c),ord='fro'))**2
        iter = iter + num_nodes
    sum = sum + lamb*np.trace(K.T@L@K)
    return sum



def optmize_k(num_elec,object_func):
    K = cp.Variable((num_elec,num_elec))
    objective = cp.Minimize(object_func)
    constraints = [K>>0, cp.diag(K) == 1]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return K