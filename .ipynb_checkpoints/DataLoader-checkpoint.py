from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import geoopt
import tqdm
from sklearn.neighbors import NearestNeighbors



class DataLoader:
    #! CRITICAL: xyz IS JUST THE PATIENTS ELECTRODES, NOT EVERYONES, THIS IS VERY IMPORTANT TO REMEMBER
    def __init__(self,limit,patient,window_size,ecogs,desired_node_iters,elecs_to_hold,xyz,k):
        self.limit = limit #max number of data points we will allow to be generated
        self.current_total_iter = 0
        self.current_node_iter = 0 #this is the indicated of how many many of this patients electrodes FOR THIS SPECFIC ELECTRODE WE HOLD OUT, have we gone thorugh
        self.patient = patient
        self.window_size = window_size
        self.desired_node_iters = desired_node_iters #desired number of datapoints we want to make per node (ie number of window slices to make)
        
        self.ecog = ecogs[patient]
        self.elecs_to_hold = elecs_to_hold #this is a list of the electrodes to hold out at some point 
        self.xyz = xyz #node locations
        self.k = k #the number of neighbors to connect together

        self.num_nodes = self.ecog.shape[1]
        self.node_held = elecs_to_hold[0] #the first electrode we are holds
        self.node_held_index = 0 
        if self.ecog.shape[0] < window_size * desired_node_iters: #makes sure we have enough data to calcuate this
            raise ValueError(f"The number of data points you want to generate:{desired_node_iters} with a window size of:{window_size} is more then we have the data to do, ecog has timelength: {self.ecog.shape[0]}")
        ####### make the patient graph useing knn #######
        num_nodes = xyz.shape[0] 
        neighbors = NearestNeighbors(n_neighbors=k).fit(xyz)
        distanceofneighbors,indicesofneighbors = neighbors.kneighbors(return_distance=True) #gets the indices of the 10 (or k) neighbors of each node

        all_edges = []
        all_edges_weights = []
        seen = set()
        for node, (nbrs, ds) in enumerate(zip(indicesofneighbors, distanceofneighbors)):
            for nbr, d in zip(nbrs, ds):
                pair = (min(node, nbr), max(node, nbr))  # canonical undirected pair
                if pair not in seen:
                    seen.add(pair)
                    all_edges.append([node, nbr])
                    all_edges_weights.append(d)
        
        self.graph_edges   = np.array(all_edges)
        self.graph_weights = np.array(all_edges_weights)


    def __iter__(self):
        return self
    

    def __next__(self):
        if self.current_total_iter >= self.limit:
            raise StopIteration
        if self.current_node_iter >= self.desired_node_iters:
            raise StopIteration
    
        window_start = self.current_node_iter * self.window_size
        window_end   = window_start + self.window_size
    
        node_features = []
        for i in range(self.num_nodes):
            if i in self.elecs_to_hold:
                node_features.append(np.zeros(self.window_size))  # zero out ALL held
            else:
                node_features.append(self.ecog[window_start:window_end, i])
    
        # targets for ALL held electrodes
        targets = np.array([
            self.ecog[window_start:window_end, i] for i in self.elecs_to_hold
        ])  # [num_held, win_size]
    
        self.current_total_iter += 1
        self.current_node_iter  += 1
    
        return {
            "features":    np.array(node_features),  # [nodes, win_size]
            "edges":       self.graph_edges,
            "weights":     self.graph_weights,
            "target":      targets,                  # [num_held, win_size]
            "locs":        self.xyz,
            "elecs_held":  self.elecs_to_hold,       # which indices were zeroed
        }

        
