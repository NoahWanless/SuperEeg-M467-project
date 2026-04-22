from scipy.stats import kurtosis
from scipy import signal
from pathlib import Path
import scipy
import numpy as np
import scipy
import pandas as pd
from scipy.io import loadmat

def car(data):
    """
    Common Average Reference
    data: (time, channels)
    returns: (time, channels)
    """
    #print("data.shape")
    #print(data.shape)
    #print("np.isnan(data)")
    #print(np.isnan(data).sum())
    #print(data.mean(axis=1, keepdims=True))
    return data - data.mean(axis=1, keepdims=True) 


def remove_duplicates(ecogs, xyz):
    """
    Remove duplicate electrodes based on xyz coordinates.
    Keeps first occurrence, removes same index from both ecogs and xyz.
    """
    xyz_arr  = np.array(xyz)   # shape: (n_electrodes, 3)

    # Get unique indices from xyz 
    _, unique_idx = np.unique(xyz_arr, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)  # preserve original order

    # Apply to xyz (numpy array)
    filtered_xyz = xyz_arr[unique_idx]

    #turn local indicies to global
    boundaries = np.cumsum([ecog.shape[1] for ecog in ecogs])
    patient_ids = np.searchsorted(boundaries, unique_idx, side='right')
    boundaries = ([0] + list(boundaries))
    filtered_ecogs = [
        ecogs[p][:, unique_idx[patient_ids == p] - boundaries[p]]
        for p in range(len(ecogs))
    ]

    return filtered_ecogs, filtered_xyz


########### get_single_registered_out: ###########
# takes in the directory path (if given) to the regester outputs and grabs all the 
# patients electrode locations
def get_electrode_normalized_loc(registered_dir = Path("../SuperEeg-M467-project/registered_outputs")):
    #locs_root = Path("../faces_basic/locs")
    npy_files = sorted(registered_dir.glob("*_xslocs_registered_mm.npy")) # Same sorted order as registration code
    ecogs_list_loc = []
    #mapping = []
    for i, npy_file in enumerate(npy_files):
        pts = np.load(npy_file)  # already in MNI space
        ecogs_list_loc.append(pts)
        
    xyz = np.vstack(ecogs_list_loc)  # 714 x 3, same order as mapping
    return xyz 

########### get_ecog_data: ###########
# registered_dir: the directory where the regestered outputs are (ie the transformed electrode locations)
# data_root: the directory where the voltage data lives
# we return ONLY THE VOLTAGE DATA, not the corresponding stimulus data
def get_just_ecog_data(registered_dir = Path("../SuperEeg-M467-project/registered_outputs"),data_root = Path("../faces_basic/data")):
    npy_files = sorted(registered_dir.glob("*_xslocs_registered_mm.npy")) # Same sorted order as registration code
    print(npy_files)
    patient_ids = [f.name.split("_")[0] for f in npy_files] # Same sorted patient order as everything else
    ecogs = []
    for pid in patient_ids:
        # find the .mat file for this patient
        mat_files = list((data_root / pid).glob("*.mat"))
        if not mat_files:
            print(f"[{pid}] no data .mat found!")
            continue
        ecog = loadmat(str(mat_files[0]))
        ecogs.append(ecog['data'])
    return ecogs

# This function clips the timeseries of the ecog data such that it removes
# the time from one patient which is very much messed up, shouldnt ruin too much to clip everything
# if you are ever wondering why its this number, run this line of code after ONLY loading the ecog data in,
# dont do anything to it:
# plt.plot(ecogs[7][256000:,10])
# and you will see why
def clip_time_series(ecogs,time_cutoff=256000):
    new_ecogs = []
    for pat in ecogs:
        new_ecogs.append(pat[:time_cutoff,:])
    return new_ecogs

# This function takes the ecogs (and whatever patient you are holding out)
# and applyes the car function with the special interaction of if we are holding out
# electrodes, then we apply the car function to the patient those data points WOULD have been in
# (but not including the held out ones) then subtract that saved mean from the held out ones too
# If you dont have any held out electrodes, dont put a held_out value in, and itll just regularlly apply the car function
def apply_car_function(ecogs,held_out=-2):
    new_ecogs = []
    for i,pat in enumerate(ecogs):
        if i == held_out: #if we are at the patient were they(held out electrodes) would in theory have been
            held_out_car = pat.mean(axis=1, keepdims=True) 
            new_ecogs.append(car(pat))
        elif i == held_out+1: #if we are at the electrode were the held out one would have been
            new_ecogs.append(pat - held_out_car)
        else: #regular patients
            new_ecogs.append(car(pat))
    return new_ecogs
            

########### preprocessing: ###########
# xyz: the output of get_electrode_normalized_loc(), or the list of normalized electrode locations
# ecogs: the voltage data for all patients, for all electrodes and time
# notch_size: the size of the notch we remove around desired frequencies
# minus_mean: subtracts the mean from the voltage (True or False)
###### RETURNS: ######
# xyz_clean: the normalized electrode locations cleaned out
# cleaned: the actual voltage data with certain electrodes removed (we use this var under the name dropped later)
def preprocessing(ecogs,xyz,notch_size,minus_mean=False):
    print("Processing Patients...")
    ######## Step one: apply the butternotch filter! ########
    sos = signal.butter(4, [59-notch_size, 60+notch_size], btype='bandstop', analog=False, 
                            output='sos', fs=1000)
    filtered = []
    for file in ecogs:
        filtered.append(signal.sosfiltfilt(sos, file,axis=0))
    ######## Step two: kurtosis ########
    cleaned = [] #the cleaned data itself
    electrode_offset = 0
    kept_global_indices = [] #global indices of electrodes kept
    for i, file in enumerate(filtered): #for each electrode we run through, get the filtered voltage and the index, and update things 
        n_electrodes = file.shape[1]
        k = kurtosis(file, axis=0)
        good_idx = np.where(k <= 10)[0] #these are local indices
        if len(good_idx) < 3: #makes sure that at least 3 electrodes exist
            electrode_offset += n_electrodes #update the offset (still needed even if skipping)
            print("Warning! a patient with less then 3 electrodes has been found and is being skipped.")
            continue
        global_good_idx = good_idx + electrode_offset #now these guys are global indices (this is for remembering what positions to keep)
        kept_global_indices.extend(global_good_idx) #concatenate them to the list
        electrode_offset += n_electrodes #update the offset

        cleaned_file = file[:, good_idx] #creates the cleaned file 
        cleaned.append(cleaned_file)
    
    ##### updating all the final elements to return #####
    kept_global_indices = np.array(kept_global_indices) #i dont think the order here matters
    xyz_clean = xyz[kept_global_indices]
    print('Preprocessing done')
    return xyz_clean, cleaned


########### hold_out: ###########
# xyz_clean: the output of get_electrode_normalized_loc(), or the list of normalized electrode locations
# cleaned: the voltage data for all patients, for all electrodes and time
# pat_hold: The patient to hold electrodes from 
# elecs_to_hold: The LIST of local electrode indices to hold out for this patient
###### RETURNS: ######
# cleaned,xyz_clean, the same as input, but with the elements moved to the needed locations
# location_offset: this is the GLOBAL index where the fake patient begans, and lasts for len(elecs_to_hold) STARTING FROM location_offset
###### Notes: ######
# This is how it works: from patient 'pat_hold' we remove the electrodes given, and then reattach then, as there own
# so called 'patient' to the list 'cleaned'. This 'patient' is ALWAYS place directly after the patient we are removing it from
# the same process is applied to the electrode locations, shuffling the indices that they live in so that they go to the right spots
# ! SUPER IMPORTANT THIS FUNCTION CAN ONLY BE RUN ONCE, BECAUSE PYTHON ASSGINS VARS BY REFERENCE I CANT GET THIS TO
# ! NOT EDIT THE ORGINAL VARAIBLES PASSED IN AT THE SAME TIME, SO TO AVOID MISTAKES NEVEERRRRRRRRR EVERRRRRRR RUN THIS MORE THEN ONCE
# ! UNLESS, YOU FIRST REDEFINE THE VARIABLES YOU ARE PASSING INTO THIS FUNCTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# ! ALSO THIS HAS NOT BEEN TEST FOR 1 ELECTRODE, SO DONT DO THAT UNTIL I TEST IT
def hold_out(xyz_data,ecog_data,pat_hold,elecs_to_hold):
    cleaned = ecog_data #this is needed so changes are not made inplace so reruns of the function dont break things
    xyz_clean = xyz_data
    ######### Find Electrode offset for this patient for global indices #########
    elec_offset = 0 #electrode offset to turn local into global indices
    for i in range(len(cleaned)): #for each patient
        if i < pat_hold:
            elec_offset+=cleaned[i].shape[1] #adds the number of electrodes
    global_elecs_to_hold = np.array(elecs_to_hold) + elec_offset 
    ######### Find all held out electrodes and remove them (from cleaned)#########
    pat = cleaned[pat_hold] #this is the version of the file we will edit and remove the electrodes from
    held_out_elcs =[] #list where we will store the held out electrode data
    for elec in elecs_to_hold:
        held_out_elcs.append(cleaned[pat_hold][:,elec]) #all timeseires for that electrode
    pat = np.delete(pat, elecs_to_hold,axis=1) #removes that electrode from the patients data
    held_out_elcs = np.array(held_out_elcs).T #to get it the right orinentation    
    cleaned[pat_hold] = pat
    cleaned.insert(pat_hold+1,held_out_elcs) #pat_hold+1 will be the index of the next patient, thus shifting everything over one
    ######### Find the location to put the electrodes and move them lists #########
    locations_held = xyz_clean[global_elecs_to_hold]
    xyz_clean = np.delete(xyz_clean, global_elecs_to_hold,axis=0) 
    location_offset = 0 #this is the location offset we will use to figure out were to put the location data now (for the position between patient pat_hold, and pat_hold+1)
    for i in range(len(cleaned)): #for each patient
        if i <= pat_hold:
            location_offset+=cleaned[i].shape[1] #adds the number of electrodes
    xyz_clean = np.insert(xyz_clean,location_offset,values=locations_held,axis=0)
    return cleaned,xyz_clean,location_offset
    

# All three inputs are part of the outputs of the full_preprocessing() function
# xyz_clean: the normalized electrode locations cleaned out
# dropped: the actual voltage data with certain electrodes removed
# mapping_clean: this one i dont know
def make_rbf_correlation_matrix(xyz_clean,dropped,mapping_clean):
    #create RBF correlation matrix
    # Euclidean distance matrix (714x714)
    #dist_matrix = cdist(xyz_clean, xyz_clean, metric='euclidean')
    dist_matrix = scipy.spatial.distance.cdist(xyz_clean, xyz_clean, metric='euclidean')
    # Gaussian RBF kernel: exp(-(epsilon * r)^2)
    rbf_matrix = np.exp(-dist_matrix**2 / 20)

    correlation_matrices = []
    for i, matrix in enumerate(dropped): 
        # Get electrode indices for this patient
        patient_electrode_indices = mapping_clean[mapping_clean[:, 1] == i, 0].astype(int)   
        # Compute pairwise correlation between this patient's electrodes
        corr = pd.DataFrame(matrix).corr() #^ COLLECT THESE FOR JUST THE PATIENT CORRELTAION MATRIX
        # shape: (n_patient_elec x n_patient_elec)
        # RBF weights between ALL 649 electrodes and this patient's electrodes
        # W shape: (714 x n_patient_elec)
        W = rbf_matrix[:, patient_electrode_indices]
        # Equation (6): Cˆ(x,y) = sum_ij W(x,i)*W(y,j)*z(C̄s(i,j))
        #                        / sum_ij W(x,i)*W(y,j)
        # Vectorized: numerator = W @ z_corr @ W.T  →  (714 x 714)
        numerator   = W @ corr @ W.T
        denominator = W @ np.ones_like(corr) @ W.T

        C_hat = np.divide(numerator, denominator, 
                            where=denominator != 0, 
                            out=np.zeros((649, 649)))

        correlation_matrices.append(C_hat)
    return correlation_matrices




# All three inputs are part of the outputs of the full_preprocessing() function
# xyz_clean: the normalized electrode locations cleaned out
# dropped: all electrodes and there timeserises (this is the cleaned dropped)
# mapping_clean: this one i dont know
# this returns the list of the patient correlation matrices but only of the size for the 
# number of electrodes for that patients
def make_patient_correlation_matrix(xyz_clean,dropped):
    #create RBF correlation matrix
    # Euclidean distance matrix (714x714)
    #dist_matrix = cdist(xyz_clean, xyz_clean, metric='euclidean')
    dist_matrix = scipy.spatial.distance.cdist(xyz_clean, xyz_clean, metric='euclidean')
    # Gaussian RBF kernel: exp(-(epsilon * r)^2)
    rbf_matrix = np.exp(-dist_matrix**2 / 20)

    correlation_matrices = []
    for i, matrix in enumerate(dropped): 
        # Get electrode indices for this patient
        #patient_electrode_indices = mapping_clean[mapping_clean[:, 1] == i, 0].astype(int)   
        # Compute pairwise correlation between this patient's electrodes
        corr = pd.DataFrame(matrix).corr() #^ COLLECT THESE FOR JUST THE PATIENT CORRELTAION MATRIX

        correlation_matrices.append(corr)
    return correlation_matrices
        





############## Tara getdata function: ##############
# This function takes in the project root dicertory, 
# the directory of the electrode locations, and the fmri brain data
# any example of the inputs might be: 
#######
#project_root = Path("/Users/rustin/Documents/Big Data 567/SuperEeg-M467-project")
#brains_root = Path("/Users/rustin/Documents/Big Data 567/faces_basic/brains")
#locs_root = Path("/Users/rustin/Documents/Big Data 567/faces_basic/locs")
#######
# then it extracts the electrode locations after being transformed and 
# saving them to a folder to be used later and more easily 
# this function ideally needs only to be run once to get the data
# !NOTE: this function uses ALL electrode locations, so some of them, if they are bad,
# ! will not be pulled out but need to be done so later, remember this!
'''
def getdata(project_root,brains_root,locs_root):
    # Single destination folder for all registered outputs
    all_registered_dir = project_root / "registered_outputs"
    all_registered_dir.mkdir(parents=True, exist_ok=True)

    brain_ids = {p.name for p in brains_root.iterdir() if p.is_dir()}
    loc_ids = {p.name.split("_")[0] for p in locs_root.glob("*_xslocs.mat")}
    patient_ids = sorted(brain_ids & loc_ids)

    print(f"Found {len(patient_ids)} shared patients: {patient_ids}")
    print(f"Saving all registered outputs to: {all_registered_dir}")

    template_img_ants = ants.image_read(ants.get_ants_data("mni")) #gets the template brain image

    for pid in patient_ids: 
        mri_path = brains_root / pid / f"{pid}_mri.nii"
        loc_path = locs_root / f"{pid}_xslocs.mat"

        if not mri_path.exists():
            print(f"[{pid}] missing MRI: {mri_path}")
            continue
        if not loc_path.exists():
            print(f"[{pid}] missing locs: {loc_path}")
            continue

        print(f"\n[{pid}] registering...")

        raw_img_ants = ants.image_read(str(mri_path), reorient="IAL")
        transformation = ants.registration(
            fixed=template_img_ants,
            moving=raw_img_ants,
            type_of_transform="SyN",
            verbose=False,
        )

        registered_img_ants = transformation["warpedmovout"] #transforms brain image
        out_mri_path = all_registered_dir / f"{pid}_mri_registered.nii" #this is the output mri data that will be made
        registered_img_ants.to_file(str(out_mri_path))
        #makes the locations a np array
        locs_mm = np.asarray(loadmat(str(loc_path), squeeze_me=True)["locs"], dtype=float).reshape(-1, 3)
        pts_df = pd.DataFrame(locs_mm, columns=["x", "y", "z"])

        # For points, ANTs mapping is opposite image mapping.
        # Native (moving) -> template (fixed) uses invtransforms here.
        pts_fixed_df = ants.apply_transforms_to_points(
            dim=3,
            points=pts_df,
            transformlist=transformation["invtransforms"],
            whichtoinvert=[True, False],
        )

        locs_fixed_mm = pts_fixed_df[["x", "y", "z"]].to_numpy(dtype=np.float64)
        out_npy_path = all_registered_dir / f"{pid}_xslocs_registered_mm.npy"
        np.save(out_npy_path, locs_fixed_mm)

        print(f"[{pid}] MRI saved to: {out_mri_path}")
        print(f"[{pid}] locs saved to: {out_npy_path}  shape={locs_fixed_mm.shape}")

    print("\nAll done.")
    '''