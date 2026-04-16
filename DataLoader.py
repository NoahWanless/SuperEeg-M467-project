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

        # turn indices lists into pairwiase combos
        all_edges = []
        all_edges_weights = []
        iter = 0 #iter is the node of which we are considering its neightbors
        for indexs,distances in zip(indicesofneighbors,distanceofneighbors):
            for num,dist in zip(indexs,distances): #the neighbors of node 'iter'
                all_edges.append([iter,num])
                all_edges.append([num,iter]) #adds the edge going the other direction 
                all_edges_weights.append(dist)#adds the distance twice, because technically two edges exist
                all_edges_weights.append(dist)
            iter += 1
        self.graph_edges = np.array(all_edges)
        self.graph_weights = np.array(all_edges_weights) 



    def __iter__(self):
        return self
    

    def __next__(self):
        #checks, have we gone over the max datapoints we want to generate
        if self.current_total_iter >= self.limit: 
            raise StopIteration
        #ie we have generated enought points for this electrode so we that we will move on to the next
        # or have we done wnough window blocks for our purposes
        if self.desired_node_iters <= self.current_node_iter: 
            #checks, if this was the last of the nodes we want to hold out, if so, end
            if self.node_held_index == len(self.elecs_to_hold)-1: 
                raise StopIteration
            #otherwise
            self.current_node_iter = 0 #reset number of datapoints we have made
            self.node_held_index +=1 #change what electrode we are dealing with (in local list of elecs_to_hold)
            self.node_held = self.elecs_to_hold[self.node_held_index] #what node we are dealing with in larger list
        
        #get the node features
        # remember: self.current_node_iter determines where in the timeseries we are when times with the window sie
        node_features = []
        node_held = None
        for i in range(self.num_nodes): #for each node
            if i !=self.node_held:  #if its one we dont hold out
                window_start = self.current_node_iter * self.window_size #make window frame
                window_end = window_start + self.window_size
                ecog_masked = self.ecog[window_start:window_end,i] #for this node, we get this window
                node_features.append(ecog_masked)
            #if we are dealing with the held out node, replace it with all zeros
            elif i == self.node_held: 
                node_features.append(np.zeros(self.window_size))

                window_start = self.current_node_iter * self.window_size #make window frame
                window_end = window_start + self.window_size
                node_held = self.ecog[window_start:window_end,i]
                

        node_features = np.array(node_features)
        self.current_total_iter +=1
        self.current_node_iter +=1
        #[node_features,self.graph_edges,self.graph_weights,node_held]
        return {"features":node_features,"edges":self.graph_edges,"weights":self.graph_weights,"target":node_held}

        
