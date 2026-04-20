import numpy as np
import random
from DataLoader import DataLoader


# gets the xyz location data for just one patient from the orginal big list
# pat: index of the patient we want
# ecogs: the ecog data
# xyz: the global location data
def get_1_patient_locations(pat,ecogs,xyz):
    elec_nums = []
    for temp in ecogs: #gets the number of electrodes for each patient
        elec_nums.append(temp.shape[1])

    pat_index_start = 0
    pat_index_end = 0
    for i,num in enumerate(elec_nums): #makes the starting and ending indices forthis patient
        if i < pat: #adds up all the patients before this one to get the starting node (inclusive)
            pat_index_start += num
        if i <= pat:#adds up all the patients before this one AND ITSELF to get the ending node (exclusive)
            pat_index_end += num
    return xyz[pat_index_start:pat_index_end]



########### Claude generated ###########
# takes in a list of iterators
def sample_iterators(iterators):
    """Randomly sample from multiple iterators until all are exhausted."""
    # Wrap each iterator to track exhaustion
    active = list(iterators)
    buffers = [None] * len(active)
    exhausted = [False] * len(active)

    def try_advance(i): #tires to pull the next value from iterator i to put in the buffer
        try:
            buffers[i] = next(active[i])
            return True
        except StopIteration:
            exhausted[i] = False
            return False

    # Prime each iterator
    live_indices = [i for i in range(len(active)) if try_advance(i)]

    while live_indices: #while we have elements to pull from
        # Pick a random live iterator
        idx = random.choice(live_indices)
        yield buffers[idx]

        # Advance it; remove if exhausted
        if not try_advance(idx):
            live_indices.remove(idx)


# ecogs_cleaned: the ecog data
# xyz_clean: the global location data for the electrodes
# win_size: the desired size of windows to take span shots from for the ecog timeseries data
# safety_size: amount of millieseconds to clip off of the end of the timeseries to ensure that the math wont break (id leave at the default)
# k: the k value for k neigherest neighbors to determine graph connectivity for the dataloader
# max_data_points: the MAX number of datapoints we will allow it to generate if you want it to clip for whatever reason
############ RETURNS: ############
# list of iterators
def built_iterator_list(ecogs_cleaned,xyz_clean,win_size,safety_size = 200,k = 10,max_data_points = 10000000):
    iterators = []
    factor_held_out = 10 #ie number held out is num_nodes / factor_held_out, rounded to nearest int, so bigger factor_held_out means less held out
    for pat_index in range(len(ecogs_cleaned)): #for each patient
        pat_locs = get_1_patient_locations(pat_index,ecogs_cleaned,xyz_clean)    
        desired_iter_per_node = ecogs_cleaned[pat_index].shape[0]/win_size - safety_size #timeseries length over window size with a little bit of wiggle room for safety
        pat_num_nodes = ecogs_cleaned[pat_index].shape[1]
        list_held = np.random.randint(low=0,high=pat_num_nodes-1,size=int(pat_num_nodes/factor_held_out))

        loader = DataLoader(limit=max_data_points,
                            patient=pat_index,
                            window_size=win_size,
                            ecogs=ecogs_cleaned,
                            desired_node_iters=desired_iter_per_node,
                            elecs_to_hold=list_held,
                            xyz=pat_locs,
                            k=k)
        iterators.append(loader)
    return iterators