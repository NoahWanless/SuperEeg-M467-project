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


########### full_preprocessing: ###########
# xyz: the output of get_electrode_normalized_loc(), or the list of normalized electrode locations
# ecogs: the voltage data for all patients, for all electrodes and time
# notch_size: the size of the notch we remove around desired frequencies
# minus_mean: subtracts the mean from the voltage (True or False)
# RETURNS:
# xyz_clean: the normalized electrode locations cleaned out
# mapping_clean #? I DONT KNOW WHAT THIS IS TARA IF YOU WANT TO ANSWER THAT THAT WOULD BE GREAT
# kept_global_indices: the indexs of what electrodes were kept 
# cleaned: the actual voltage data with certain electrodes removed (we use this var under the name dropped later)
# patient_electrode_info: how many electordes each patient has recorded for use (#!CURRENTLY NOT BEING RETURNED)
def full_preprocessing(ecogs,xyz,notch_size,minus_mean=False):

    #Step one: apply the butternotch filter!
    sos = signal.butter(4, [59-notch_size, 60+notch_size], btype='bandstop', analog=False, 
                            output='sos', fs=1000)
    filtered = []
    for file in ecogs:
        filtered.append(signal.sosfiltfilt(sos, file))
    #Step two: kurtosis
    cleaned = []
    kept_global_indices = []
    mapping_clean = []
    electrode_offset = 0
    patient_electrode_info = []
    for i, file in enumerate(filtered): #for each electrode we run through, get the filtered voltage and the index
        n_electrodes = file.shape[1]
        k = kurtosis(file, axis=0)
        good_idx = np.where(k <= 10)[0]
        
        cleaned_file = file[:, good_idx]
        if minus_mean:
            cleaned.append(car(cleaned_file)) #subtracts the mean
        else:
            cleaned.append(cleaned_file)
        
        global_good_idx = good_idx + electrode_offset
        kept_global_indices.extend(global_good_idx)
        electrode_offset += n_electrodes
        patient_electrode_info.append(len(good_idx)) #records the number of kept electrodes
        for j in range(len(good_idx)):  # number of KEPT electrodes
            mapping_clean.append([len(mapping_clean), i])  # i = subject index

    mapping_clean = np.array(mapping_clean)
    kept_global_indices = np.array(kept_global_indices) #!THIS MUST GET RETURNED FOR FUTURE USE
    xyz_clean = xyz[kept_global_indices]

    #print(xyz_clean.shape)        # (n_kept_electrodes, 3)
    #print(mapping_clean.shape)    # (n_kept_electrodes, 2)
    
    return xyz_clean, mapping_clean, kept_global_indices, cleaned
    #step three is remove patients with 2 electrodes or less, but we already know they all have way more then that
    

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



#TODO: MAKE A VERSION THAT ADDS A NEW 'PATIENT, AND THIS ONE WILL BE FAKE'
#TODO: MAKE A VERSION THAT ADDS A NEW 'PATIENT, AND THIS ONE WILL BE FAKE'

########### full_preprocessing: ###########
# xyz: the output of get_electrode_normalized_loc(), or the list of normalized electrode locations
# ecogs: the voltage data for all patients, for all electrodes and time
# notch_size: the size of the notch we remove around desired frequencies
# minus_mean: subtracts the mean from the voltage (True or False)
# pat_to_hold: the patient 0-13, to hold out the final electrode from to use for validation
# RETURNS:
# xyz_clean: the normalized electrode locations cleaned out
# mapping_clean: ??? 
# kept_global_indices: the indexs of what electrodes were kept 
# cleaned: the actual voltage data with certain electrodes removed (we use this var under the name dropped later)
# patient_electrode_info: how many electordes each patient has recorded for use (#!CURRENTLY NOT BEING RETURNED)
# hold_out_file: this is the electrode info of the held out electrode, taken directly
#! NOTE THIS IS FROM THE UNFILTED DATA, NOT THE FILTERED ONES I THINK?
def full_preprocessing_hold(ecogs,xyz,notch_size,minus_mean=False,pat_to_hold=-1):
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
    kept_global_indices = [] #global indices of electrodes kept
    mapping_clean = []
    electrode_offset = 0
    patient_electrode_info = []
    for i, file in enumerate(filtered): #for each electrode we run through, get the filtered voltage and the index, and update things 
        print(f"Looking at patient: {i}")
        n_electrodes = file.shape[1]
        k = kurtosis(file, axis=0)
        good_idx = np.where(k <= 10)[0] #these are local indices
        ###### if we are holding out, record the electrode index, and the file
        ###### for that one, along with removing the index from the big list
        if pat_to_hold == i: 
            elec_to_hold = good_idx[(len(good_idx) - 1)] #gets the last of the remaining element to remove (the index of it locally to this patient)
            #kurtosis_diff = n_electrodes - len(good_idx) #this is something that was here and i dont know what it was for
            hold_out_file = file[:, elec_to_hold]
            good_idx = np.delete(good_idx, (len(good_idx) - 1))
            
        ####### updating the indexs kept globally #######
        global_good_idx = good_idx + electrode_offset #now these guys are global indices
        kept_global_indices.extend(global_good_idx)

        ######## Appendimg to the list of global indices ########
        if pat_to_hold == i: 
            kept_global_indices.append(elec_to_hold + electrode_offset)  #this addes the global index of the held out one onre
            global_held = (elec_to_hold + electrode_offset) #the global index of the held out thing but of the orginal dataset, not the one with all these things removed(via kurtosis)

        ##### updating patient electrode info #####
        electrode_offset += n_electrodes
        patient_electrode_info.append(len(good_idx)) #records the number of kept electrodes
        for j in range(len(good_idx)):  # number of KEPT electrodes, add what ones were kept
            if i < pat_to_hold:
                mapping_clean.append([len(mapping_clean), i])  # i = subject index    
            elif i == pat_to_hold:
                mapping_clean.append([len(mapping_clean), i])  # i = subject index    
            else:
                mapping_clean.append([len(mapping_clean), i+1])        

        cleaned_file = file[:, good_idx] #creates the cleaned file 

        if minus_mean:
            cleaned.append(car(cleaned_file)) #subtracts the mean if desired
        else:
            cleaned.append(cleaned_file)
        
        ##### This does the final addition of the hold out file to the list of cleaned files
        if pat_to_hold == i:
            patient_electrode_info.append(1) #also adds the fakes patients info to the thing
            mapping_clean.append([len(mapping_clean), i+1]) #this appends the fake patient info to the mapping clean data
            if minus_mean:
                cleaned.append(car(hold_out_file)) #subtracts the mean
            else: #these two require a extra dimension for consistancy with everything
                #TODO: how many elements are in cleaned, and that would give the local index
                local_index_held = len(cleaned)
                cleaned.append(hold_out_file)
                
        
            
    ##### updating all the final elements to return #####
    mapping_clean = np.array(mapping_clean)
    kept_global_indices = np.array(kept_global_indices) 
    xyz_clean = xyz[kept_global_indices]
    
    return xyz_clean, mapping_clean, kept_global_indices, cleaned, hold_out_file


#TODO: MAKE A VERSION THAT ADDS A NEW 'PATIENT, AND THIS ONE WILL BE FAKE'
#TODO: MAKE A VERSION THAT ADDS A NEW 'PATIENT, AND THIS ONE WILL BE FAKE'

# All three inputs are part of the outputs of the full_preprocessing() function
# xyz_clean: the normalized electrode locations cleaned out
# dropped: all electrodes and there timeserises (this is the cleaned dropped)
# mapping_clean: this one i dont know
# this returns the list of the patient correlation matrices but only of the size for the 
# number of electrodes for that patients
def make_patient_correlation_matrix(xyz_clean,dropped,mapping_clean):
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