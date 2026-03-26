from scipy.stats import kurtosis
from scipy import signal
from pathlib import Path
import scipy
import ants
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
    return data - data.mean(axis=0, keepdims=True) 


########### get_single_registered_out: ###########
# takes in the directory path (if given) to the regester outputs and grabs all the 
# patients electrode locations
def get_electrode_normalized_loc(registered_dir = Path("../SuperEeg-M467-project/registered_outputs")):
    registered_dir = Path("../SuperEeg-M467-project/registered_outputs")
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
    registered_dir = Path("../SuperEeg-M467-project/registered_outputs")
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

########### full_preprocessing_hold: ###########
# xyz: the output of get_electrode_normalized_loc(), or the list of normalized electrode locations
# ecogs: the voltage data for all patients, for all electrodes and time
# notch_size: the size of the notch we remove around desired frequencies
# minus_mean: subtracts the mean from the voltage (True or False)
# pat_to_hold: One of two meanings:
#       pat_to_hold = -1: this means we are not holding out any electrodes
#       pat_to_hold = 0-13: this tells what patient to hold out a electrode from
# elec_hold_seed: one of 3 meanings:
#       elec_hold_seed = -1: we hold out the last electrode
#       elec_hold_seed = -2: we hold out a random electrode from the patient
#       elec_hold_seed = a generic value: for reproducable results, this is the index to pick what electrode to hold out,
#                                         generally a value between 0-20 ish depending on how many the patient has.
###### RETURNS: ######
# xyz_clean: the normalized electrode locations cleaned out
# cleaned: the actual voltage data with certain electrodes removed (we use this var under the name dropped later)
# hold_out_file: this is the electrode info of the held out electrode, taken directly
# global_held_index: the index of the held out electrode globally in the cleaned out data
def full_preprocessing_hold(ecogs,xyz,notch_size,minus_mean=False,pat_to_hold=-1,elec_hold_seed=-1):
    ######## Step one: apply the butternotch filter! ########
    sos = signal.butter(4, [59-notch_size, 60+notch_size], btype='bandstop', analog=False, 
                            output='sos', fs=1000)
    filtered = []
    num_elec_pat = []
    for file in ecogs:
        num_elec_pat.append(len(file))
        filtered.append(signal.sosfiltfilt(sos, file))
    print("Filter applied")
    ######## Step two: kurtosis, and holding out ########
    cleaned = [] #the cleaned data itself
    electrode_offset = 0
    kept_global_indices = [] #global indices of electrodes kept
    for i, file in enumerate(filtered): #for each electrode we run through, get the filtered voltage and the index, and update things 
        print(f"Looking at patient: {i}")
        n_electrodes = file.shape[1]
        k = kurtosis(file, axis=0)
        good_idx = np.where(k <= 10)[0] #these are local indices
        ###### if we are holding out, record the electrode index, and the file
        ###### for that one, along with removing the index from the big list
        if pat_to_hold == i: 
            if elec_hold_seed == -1:
                elec_to_hold = good_idx[(len(good_idx) - 1)] #gets the last of the remaining element to remove (the index of it locally to this patient)
                ind_of_hold = (len(good_idx) - 1)
            elif elec_hold_seed == -2: #^ this is random seeding, so we choose a random value
                rand_ind = np.random.randint(0, len(good_idx)) #the high value is exculsive 
                elec_to_hold = good_idx[rand_ind]
                ind_of_hold = rand_ind
            else:#^ this is manuel choosing of the elec to hold out
                if elec_hold_seed < 0 or elec_hold_seed > (len(good_idx) - 1):
                    raise ValueError("Your 'elec_hold_seed' value is out of range, try something else, mnore then likely a smaller value")
                elec_to_hold = good_idx[elec_hold_seed] 
                ind_of_hold = elec_hold_seed #this is the index (from the good_idx array) of the thing we are holding, so we are remvoing it from the good_idx array
        
            hold_out_file = file[:, elec_to_hold]
            good_idx = np.delete(good_idx, ind_of_hold)
        
        global_good_idx = good_idx + electrode_offset #now these guys are global indices (this is for remembering what positions to keep)
        kept_global_indices.extend(global_good_idx)
        if pat_to_hold == i: 
            kept_global_indices.append(elec_to_hold + electrode_offset)
        electrode_offset += n_electrodes

        cleaned_file = file[:, good_idx] #creates the cleaned file 
        if minus_mean:
            cleaned.append(car(cleaned_file)) #subtracts the mean if desired
        else:
            cleaned.append(cleaned_file)
        
        ##### This does the final addition of the hold out file to the list of cleaned files
        if pat_to_hold == i:
            global_held_index = 0
            for pat in cleaned: #gets the total num of electrodes so far, this will be our index for the held out file we will add
                global_held_index += pat.shape[1] #this is to return 
            if minus_mean:
                cleaned.append(car(hold_out_file)) #subtracts the mean
                hold_out_file = car(hold_out_file)
            else: #these two require a extra dimension for consistancy with everything
                cleaned.append(hold_out_file)
                
        

    ##### updating all the final elements to return #####
    kept_global_indices = np.array(kept_global_indices) #i dont think the order here matters
    xyz_clean = xyz[kept_global_indices]
    #! global_held_index, switch to returning this if you want to get the global index of the held out file
    return xyz_clean, cleaned, hold_out_file, global_held_index


    

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