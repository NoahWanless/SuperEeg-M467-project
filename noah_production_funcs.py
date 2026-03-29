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
import geoopt



"""
from tara_preprocessing import full_preprocessing,make_rbf_correlation_matrix,get_just_ecog_data,get_electrode_normalized_loc,make_patient_correlation_matrix

data_root = Path("/Users/noahwanless/Desktop/Spring2026/M467/faces_basic/data")
registered_dir = Path("../SuperEeg-M467-project/registered_outputs")
ecogs = get_just_ecog_data(registered_dir,data_root)
xyz = get_electrode_normalized_loc(registered_dir)
xyz_clean, mapping_clean, kept_global_indices, cleaned = full_preprocessing(ecogs,xyz)
patient_corr_mat = make_patient_correlation_matrix(xyz_clean,cleaned,mapping_clean)

from noad_production_funcs import create_u
U_det, loss = create_u(k=10,r=100,lamb=1,patient_corr_mat=patient_corr_mat,xyz_clean=xyz_clean,object_func=object_func)

"""
# above would be good example piece of code of how to load in all the needed data and prepare for using create_u function
# remember that the paths will be different for you
# and also remember that to get the full U you would need to do (U_det@U_det.T)



####### create_lapaican_knn #######
# xyz_clean: the points to use to make the graph
# k: how many nearest neighbors to connect points to
####### Returns: #######
# Glaplacian, the laplacian of the matrix in array form
def create_lapaican_knn(xyz_clean,k):
    ############## Make Graph ##############
    num_nodes = xyz_clean.shape[0] #649
    neigh = NearestNeighbors(n_neighbors=k).fit(xyz_clean)
    indicesofneigh = neigh.kneighbors()[1] #gets the indices of the 10 (or k) neighbors of each node
    # turn indices lists into pairwiase combos
    all_edges = []
    iter = 0
    for indexs in indicesofneigh:
        for num in indexs:
            all_edges.append((iter,num))
        iter += 1
    G = nx.Graph()
    nodes = np.arange(num_nodes)
    G.add_nodes_from(nodes)
    G.add_edges_from(all_edges)
    ############## Preparing function inputs ##############
    Glaplacian = nx.linalg.laplacian_matrix(G).toarray()
    return Glaplacian

#this is the rbf distance function from the paper
def rbf_dist(x,nu,lamb):
    return  np.exp(-1*(np.linalg.norm(x-nu)**2)/lamb)
    
####### create_lapaican_rbf #######
# xyz_clean: the points to use to make the graph
# lamb: scalling factor for the rbf function, see its definition
####### Returns: #######
# Glaplacian, the laplacian of the matrix in array form
def create_lapaican_rbf(xyz_clean,lamb):
    num_nodes = xyz_clean.shape[0]
    nodes = np.arange(num_nodes)
    G = nx.Graph()
    G.add_nodes_from(nodes) #adds all the nodes
    ######### Now connecting the nodes #########
    all_edges = []
    iter = 0
    for x,p1 in zip(nodes,xyz_clean):
        for y,p2 in zip(nodes,xyz_clean):
            if x!=y: #this way nodes are not connected to itself
                G.add_edge(x,y,weight=rbf_dist(p1,p2,lamb)) 
    
    Glaplacian = nx.linalg.laplacian_matrix(G,weight='weight').toarray()
    return Glaplacian






# This is the objective function defined by Javier, so if you have any questions go to him first
def object_func(C,U,L,lamb,patient_node_num):
    sum = torch.zeros(1,requires_grad=True) 
    iter = 0
    for i in range(14):
        c = C[i] #each patient correlation matrix
        num_nodes = patient_node_num[i]
        u = U[iter:iter+num_nodes,:] #all columns of rows iter+num of nodes + 1 
        sum = sum + (torch.linalg.norm((u@u.T - c),ord='fro'))**2 #+lamb*torch.trace(U.T@L@U)
        iter = iter + num_nodes
    sum = sum + lamb*torch.trace(U.T@L@U)
    return sum

############## reate_u ################ creates the U matrix
# r: the 'complexity' of our approximation (the number of columns of our U matrix we will make)
# k: 
#       the number of nearest neighbors a electrode is 'connected to' if graph='knn'
#       the scaler for the rbf function if graph='rbf'
# lamb: the parameter on trace aspect of the loss function
# xyz_clean: normalized electrode locations on the brain
# patient_corr_mat: the list of indivdual patient correlation matrices (ONLY containing the nodes they obsevered on them)
# object_func: the objective function we want to minimize
# training_steps: number of steps to train the function (usually 500 should be enough, defaults to 1000)
# lr: learning rate of the optimizer
# graph: 'knn' or 'rbf' defines what graph set up to use for making the laplacian
######### Returns #########
# U: this is the big U matrix, to get our correlation matrix do U@U.T 
# Loss: this is the list of loss at each step of training to ensure that the function is converging
######### Notes: #########
# It must be noted that yes, to our knowlegde the constrained of the manifold is applied to the rows of the 
# matrix U. as is desired by the formula
# This is checked by both a test of the output of the model (which gives a matrix were only the rows are norm 1)
# and by examining the source code of the geoopt.optim.RiemannianAdam().step() function, which shows it iterates through the 
# entires U (the only param 'group' we gave) which means iterating through the rows, and those are what is constrained to the 
# manifold
# For more consult the following sites:
# https://geoopt.readthedocs.io/en/latest/_modules/geoopt/optim/radam.html#RiemannianAdam
# https://github.com/pytorch/pytorch/blob/v2.10.0/torch/optim/optimizer.py#L342
##############################
def create_u(k,r,lamb,patient_corr_mat,xyz_clean,object_func=object_func,training_steps=1000,lr=0.01,graph='knn'):
    ############## Make Graph ##############
    num_nodes = xyz_clean.shape[0]
    if graph == 'knn':
        Glaplacian = create_lapaican_knn(xyz_clean,k)
    elif graph == 'rbf':
        Glaplacian = create_lapaican_rbf(xyz_clean,k)
    ############## Preparing function inputs ##############
    L = torch.tensor(Glaplacian,dtype=torch.float32,requires_grad=False)
    patient_node_num = [] #number of electrodes each patient has
    C = []
    for corr in patient_corr_mat:
        C.append(torch.tensor(np.array(corr),requires_grad=True))
        patient_node_num.append(corr.shape[0])
    ############## Preparing U and its manifold ##############
    rng = np.random.default_rng()
    U_intial = rng.uniform(1,2,(num_nodes,r)) #random values to start with
    U_intial = U_intial/np.linalg.norm(U_intial,axis=1,keepdims=True) #ensuring intial U is normalized
    U_tensor = torch.tensor(U_intial,dtype=torch.float32) 
    sphere = geoopt.manifolds.Sphere()
    #https://geoopt.readthedocs.io/en/latest/manifolds.html?highlight=sphere#geoopt.manifolds.Sphere
    # ^ see the above site for more on the manifolds^
    U = geoopt.ManifoldParameter(U_tensor,manifold=sphere) # makes it so the U is contrained to the sphere
    ############## Training U ##############
    optimizer = geoopt.optim.RiemannianAdam([U], lr=0.01) # adam optmizer that is aware we are stuck on the sphere
    loss_list = []
    grads = []
    print("Optimizing U")
    for step in tqdm(range(training_steps)):
        optimizer.zero_grad()
        z = object_func(C,U,L,lamb,patient_node_num) #this is our loss function
        loss_list.append(z.detach())
        z.backward()
        optimizer.step()
    return U.detach(),loss_list



"""
from tara_preprocessing import full_preprocessing,make_rbf_correlation_matrix,get_just_ecog_data,get_electrode_normalized_loc
correlation_matrix = np.load("/Users/noahwanless/Desktop/Spring2026/M467/gitproject/SuperEeg-M467-project/correlation_matrix.npy")
#^ above is the BIG correlation matrix of all the patients based on some code Tara made
data_root = Path("/Users/noahwanless/Desktop/Spring2026/M467/faces_basic/data")
registered_dir = Path("../SuperEeg-M467-project/registered_outputs")
ecogs = get_just_ecog_data(registered_dir,data_root)
xyz = get_electrode_normalized_loc(registered_dir)
xyz_clean, mapping_clean, kept_global_indices, cleaned = full_preprocessing(ecogs,xyz) #this fully preprocesses the data

from noad_production_funcs import single_patient_prediction

pred,y_real = single_patient_prediction(0,5,15,ecogs,correlation_matrix)
"""
#above is a good example of how to run the single_patient_prediction function


##########################
# patient: What patient this is 0-13 or higher if its the special hold one out situation
# ecogs: the ecog data
# correlation_matrix: the correlation matrix of electrodes across all patients
##########################
# Calculates a predication for all the other electrodes across all timesteps for all electrodes we did not observe in the
# given patient
#! NOTE: it predicts the Z score of the voltage data and the Z scored values is whats returned for comparision
# Additionally we went over in class this function, and we think its correct, it may not be but for now
# unless you see a glaring error, you can assume it works correctly
# note this is the version that does NOT hold anything out of the function and calcuations
########### Returns: ###########
# indices_we_pred: the global indices of what electrodes we are predicting
# 
def single_patient_prediction_pure(patient,ecogs,correlation_matrix):
    #########he correlation of the observed and unobserved datapoints #########
    Y = ecogs[patient] #gets this paitents data
    row_means = np.mean(Y, axis=0, keepdims=True)
    row_stds = np.std(Y, axis=0, keepdims=True)
    Y_z_score = (Y - row_means) / row_stds #turns them into there z_score for each value in the data
    ######################## this gets everything for this patient ########################
    if torch.is_tensor(correlation_matrix):
        correlation_matrix = correlation_matrix.numpy()
    patient_node_start = 0  #these are where this patients electrodes would start and end in the correlation matrix
    patient_node_end = -1 #!THESE ARE INCLUSIVE VALUES, BOTH OF THEM
    indices_we_pred = []
    for i,pat in enumerate(ecogs):
        if i < patient:
            if pat.ndim == 2:
                patient_node_start += pat.shape[1]
            else:
                patient_node_start += 1
        if i <=patient:
            if pat.ndim == 2:
                patient_node_end += pat.shape[1]
            else: #if this is the fake patient, just add 1
                patient_node_end += 1
    
    for i in range(correlation_matrix.shape[0]): #for each electrode
        if i < patient_node_start or i > patient_node_end:
            indices_we_pred.append(i) #these are the ones we are prediciting
    ################ Building the resources to actually use the forumla ################
    K_patient = correlation_matrix[:,patient_node_start:patient_node_end+1] #this gets all the electrodes that the patient has with their own correlation and that of others
    Kalpha_alpha = K_patient[patient_node_start:patient_node_end+1,:]
    Kalpha_alpha_inv = np.linalg.inv(Kalpha_alpha) 
    if patient_node_start == 0: #if the patient is the first (and thus all the nonobserved nodes are 'below' the observed ones)
        Kbeta_alpha = K_patient[patient_node_end+1:,:] # get all rows the patient doesnt have observed 
    else:# else the patient is not the first (and thus all the nonobserved nodes are both 'below' and 'above' the observed ones in the correlation matrix)
        Kbeta_alpha_1 = K_patient[0:patient_node_start,:]
        Kbeta_alpha_2 = K_patient[patient_node_end+1:,:]
        Kbeta_alpha = (np.vstack((Kbeta_alpha_1,Kbeta_alpha_2)))
    Y_patient = Y_z_score[:,0:patient_node_end+1]
    Yt = Y_patient.T
    ############# The formula #############
    pred = ((Kbeta_alpha@Kalpha_alpha_inv)@Yt).T
    return pred,indices_we_pred


# Takes in the ecogs and then converts all of it to z-score
def convert_to_z_score(ecogs):
    new_ecogs = []
    for pat in ecogs:
        row_means = np.mean(pat, axis=0, keepdims=True)
        row_stds = np.std(pat, axis=0, keepdims=True)
        z_pat = (pat - row_means) / row_stds
        new_ecogs.append(z_pat)
    return new_ecogs

#iterates though the ecog data for electrodes via a global electrode index
# returns that electrodes information from the list
#! NOTE: if you are using this remember that the car function that is applied potientally in preprocessing isnt automatically applied here
def index_ecog_globally(ecog,index):
    index_sum = 0
    for pat in ecog:
    
        if pat.ndim == 1: #if we axre dealing with the special fake patient, add a dim for iteartion reasons
            pat_t = np.array([pat])
        else: #else take the transpose so we can iterate over electrodes 
            pat_t = (pat.T) 

        for elec in pat_t: #iterate through the electrode dimension
            if index_sum == index: #if the sum of the indices we have dealt with is what we want
                if pat_t.shape[0] == 1: #check if the patietn we are iterating over is the fake one, and just return the whole thing
                    return pat 
                else: #else return just the electrode we want
                    return elec
            else: #increase the num of electrodes we have gone through
                index_sum = index_sum + 1
            

def u_metric_display(U_det,cleaned,file_held,loss):
    pred,indices = single_patient_prediction_pure(0,cleaned,(U_det@U_det.T))
    row_means = np.mean(file_held, axis=0, keepdims=True)
    row_stds = np.std(file_held, axis=0, keepdims=True)
    z_held = (file_held - row_means) / row_stds
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(U_det@U_det.T,ax=ax[0], cmap='coolwarm')  #[44:84,44:84]-patient_corr_mat[1].to_numpy()
    ax[0].set_title('Final U correlation matrix')
    ax[1].plot(loss)
    ax[1].set_title('Loss')
    plt.show()
    
    plt.figure(figsize=(16, 7))
    plt.title('Prediction vs True Values')
    plt.plot(pred[9000:9700,0],label='pred')
    plt.plot(z_held[9000:9700],label='true')
    plt.legend()
    plt.show()
    print("Here is the correlation between the predicited and actual electrode voltage values:")
    print(np.corrcoef(pred[:,0],z_held))







""" this is left over stuff
num_nodes = xyz_clean.shape[0] #649
    neigh = NearestNeighbors(n_neighbors=k).fit(xyz_clean)
    indicesofneigh = neigh.kneighbors()[1] #gets the indices of the 10 (or k) neighbors of each node
    # turn indices lists into pairwiase combos
    all_edges = []
    iter = 0
    for indexs in indicesofneigh:
        for num in indexs:
            all_edges.append((iter,num))
        iter += 1
    G = nx.Graph()
    nodes = np.arange(num_nodes)
    G.add_nodes_from(nodes)
    G.add_edges_from(all_edges)
    Glaplacian = nx.linalg.laplacian_matrix(G).toarray()

"""

