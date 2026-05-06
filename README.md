# SuperEeg-M467-project
Project for M467/M567 class on based on the SuperEeg paper using Ecog data


If you are searching for something go to the following places:

single patient prediction: go to 'noad_production_funcs.py'

new method of making U matrix: go to 'noad_production_funcs.py'

preprocessing methods: either 'tara_preprocessing.py' <- (go here first, this has the better and more commonly used methods)






## Preprocessing:
There are several steps in the preprocessing process, not all are necessary, but most are. 
If you would like to see the implemenation of these functions go to the 'tara_preprocessing.py' file. There you will find all the functions used in the below processes


Here is a overview of what each step in the process does:

''
data_root = Path("/Users/noahwanless/Desktop/Spring2026/M467/faces_basic/data")
registered_dir = Path("../SuperEeg-M467-project/registered_outputs")
ecogs = get_just_ecog_data(registered_dir,data_root)
xyz = get_electrode_normalized_loc(registered_dir)
''

The above steps load in the nii normalized brain locations (from the registered_outputs directory), and the ecog data itself (taken from the faces_basic project) these have the following shapes:
ecogs.shape = (patients,time,number of electrodes)
xyz.shape = (total number of electrodes, 3)



''
ecogs = clip_time_series(ecogs)
ecogs_no_dups,xyz_no_dups = remove_duplicates(ecogs,xyz)
''

Next the timeseries of the ecog data is clipped, this is because towards the end of one of the patients the ecog data becomes very choatic and clearly wrong. This is patient 7 (remembering we start the numebring at 0) around timestep 256000 or so. To avoid this messing with things we simply cut ALL patients so that they have no data beyond this timestep to avoid potential issues.
Additionaly the 'remove_duplicates' function removes electrodes (both their location and ecog data) where they are, by some mistake or glitch, in the same location.


''
xyz_clea, cleane = preprocessing(ecogs_no_dups,xyz_no_dups,notch_size=.05)
cleaned_f,xyz_f,fake_pat_beginning,held_out_elcs = hold_out(xyz_clea,cleane,0,[40,41])
''

Next we do the actual preprocessing, this has 2 steps, and go look at the actual function implementation for both of these for a better overview.
    i) A butterworth notch filter is applied.
    ii) A Kurtosis filter, with k=10 is applied, removing all electrodes and there ecog data that doesnt fit the k value

Then, with the data itself prepared, next we hold out some electrodes, we decide the patient and the electrodes, that ecog data is removed and is made into its own fake 'patient' this alters the shape of ecogs now to  ecogs.shape = (patients+1,time,number of electrodes) where the extra patient is held out. This ensures for when we are making predictions with a particular patient, we have electrodes with a decernable 'truth' to them, because they already belong to that patient and thus comparing the prediced verses actual works.



''
cleaned_f = apply_car_function(cleaned_f,0)
patient_corr_mat = make_patient_correlation_matrix(xyz_f,cleaned_f)
''

Finally car, a normalization function is applied (we tell it what patient we held out for technical reasons) and then create the INDIVIDUAL patient correlation matrixs to use in training later on.