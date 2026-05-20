# SuperEeg-M467-project
Project for M467/M567 class on based on the SuperEeg paper using Ecog data

There are several files containing a collection of functions in this project, here are what they contain.

* noah_production_funcs_2.py

This contains functions meant to support the use of the DataLoader class for use in training.

* noah_production_funcs_1.py

This contains most of the functions for things unrelated to preprocessing. Including objective functions, the creation and training of the U and K matrix's, the creation of graphs of the brain based on 'knn' of 'rbf' methods, prediction methods when using the full correlation matrix and functions to project down to certain subspaces. For more information go check those out

* DataLoader.py

This is the class implementaion of the DataLoader object which is a iterator meant to effective feed the gnn models datapoints that are partially precomputed in a time effective manner.

* tara_preprocessing.py

All preprocessing functions.

* helpers.py

These are functions that Rusty made for his code in the Registaration_Example.ipynb Notebook, go see that notebook for more.

These are files that contain functions that are used in the notebooks that actually bring things together

## Preprocessing:
There are several steps in the preprocessing process, not all are necessary, but most are. 
If you would like to see the implemenation of these functions go to the 'tara_preprocessing.py' file. There you will find all the functions used in the below processes


Here is a overview of what each step in the process does:


data_root = Path("/Users/noahwanless/Desktop/Spring2026/M467/faces_basic/data")\
registered_dir = Path("../SuperEeg-M467-project/registered_outputs")\
ecogs = get_just_ecog_data(registered_dir,data_root)\
xyz = get_electrode_normalized_loc(registered_dir)



The above steps load in the nii normalized brain locations (from the registered_outputs directory), and the ecog data itself (taken from the faces_basic project) these have the following shapes:\
ecogs.shape = (patients,time,number of electrodes)\
xyz.shape = (total number of electrodes, 3)\

You may have to change the file path for the faces_basic data, as that is not included in this repo


ecogs = clip_time_series(ecogs)\
ecogs_no_dups,xyz_no_dups = remove_duplicates(ecogs,xyz)


Next the timeseries of the ecog data is clipped, this is because towards the end of one of the patients the ecog data becomes very choatic and clearly wrong. This is patient 7 (remembering we start the numebring at 0) around timestep 256000 or so. To avoid this messing with things we simply cut ALL patients so that they have no data beyond this timestep to avoid potential issues.
Additionaly the 'remove_duplicates' function removes electrodes (both their location and ecog data) where they are, by some mistake or glitch, in the same location.



xyz_clea, cleane = preprocessing(ecogs_no_dups,xyz_no_dups,notch_size=.05)\
cleaned_f,xyz_f,fake_pat_beginning,held_out_elcs = hold_out(xyz_clea,cleane,0,[40,41])


Next we do the actual preprocessing, this has 2 steps, and go look at the actual function implementation for both of these for a better overview.\
    i) A butterworth notch filter is applied.\
    ii) A Kurtosis filter, with k=10 is applied, removing all electrodes and there ecog data that doesnt fit the k value

Then, with the data itself prepared, next we hold out some electrodes, we decide the patient and the electrodes, that ecog data is removed and is made into its own fake 'patient' this alters the shape of ecogs now to  ecogs.shape = (patients+1,time,number of electrodes) where the extra patient is held out. This ensures for when we are making predictions with a particular patient, we have electrodes with a decernable 'truth' to them, because they already belong to that patient and thus comparing the prediced verses actual works.




cleaned_f = apply_car_function(cleaned_f,0)\
patient_corr_mat = make_patient_correlation_matrix(xyz_f,cleaned_f)


Finally car, a normalization function is applied (we tell it what patient we held out for technical reasons) and then create the INDIVIDUAL patient correlation matrixs to use in training later on.


## Various methods:

We used vairous methods to try and make predictions and train things. For a more detailed explaination go to the code itself and there files. Here im just going to give the formulas for the training methods and key differences between them.



 * Method 1:

 $$\textup{Min}_{u}(\sum_{i=1}^{n}\lVert u_{i}u_{i}^T - c_{i} \lVert_{F}^2 + \lambda \textup{Trace}(U^TLU))$$

This is the method where our final correlation matrix is $$UU^T$$ , so keep that in mind.

$$u_{i}$$ is a chunk of U, namely for the ith patient who has say N number of electrodes, $$u_{i}$$ is the N rows of U corresponding to that patient, going in order through them. So for patient 0 they have the first N rows, then the next chunk is for patient 1 for however many electrodes they have and so forth.
This uses the forbius method and L is the Laplacian of the graph for the all the brains.

What is $$c_{i}$$ : 

It is the correlation matrix of Patient i with its own electrodes

Subject to the fact that $$\lVert u_{i}\lVert = 1$$ for all $$u_{i}$$


 * Method 2 AND the cvxpy method:
   
Both use the same formula, the cvxpy method implements the constrains directly while method 2 projects K to be on the subspaces of both constrains after each training step.

 $$\textup{Min}_{K}(\sum_{i=1}^{S}\lVert k_{i,i} - c_{i} \lVert_{F}^2 + \lambda \textup{Trace}(K^TLK))$$

 

What is $$k_{i,i}$$ : 

If K is a big matrix, our desired correlation matrix, think of it as a 2D matrix, where its rows and columns are the patients and its entires are the correlation matrixs corresponding to them. Or, row one is all patient 0's correlation matrices with electrodes from all patients (the 0th column would be patient 0's own electrodes, thus a correlation matrix we know, column 1 would be the correlation matrix of patient zero with electrodes from patient 1 etc and etc) $$k_{i,i}$$ is then the diagonal of all these pieces

This uses the forbius method (the little F subscript) and L is the Laplacian of the graph for the all the brains

What is $$c_{i}$$ : 

It is the correlation matrix of Patient i with its own electrodes

$$\lambda$$ is a tuneable parameter. Think of it as a measure of how much do we want to focus on making the patients correlation with its on electrodes as similar as can be to the truth value from the patients own individual correlation matrices.

All this is with the following constrains:
 
 K is Positive Semi Definite

 and the Diagonal of K is all 1's


 * Prediction

This is dont by a formula from Lucys paper, to see its impelementation go look for 'patient_prediction_pure' in 'noah_production_funcs_2.py'

$$ Y_{S,\beta} = ((\hat{K}_{\beta,\alpha} \hat{K}_{\alpha,\alpha}^{-1}) * Y_{\alpha}^T) $$


This is where $$\hat{K}_{\alpha,\alpha}$$ is the big correlation matrix but trimmed such that its only electrodes owned by patient S for both the columns and rows of K. (we'll take the inverse of this matrix)

 $$\hat{K}_{\beta,\alpha}$$ is same as above, but now we take ALL rows of the big correlation matrix, but only of the columns that are the electrodes of patient S, (ie the $$\alpha$$ part of the correlation matrix)

 Put better. $$\alpha$$ is all electrodes of the patient S, while $$\beta$$ is every electrode BUT the ones owned by patient S, and $$\hat{K}_{\alpha,\alpha}$$ or the version with $$\beta$$ is the part of the correlation matrix constrained by which electrodes we are talking about.

Finally 
$$Y_{\alpha}$$ is the ecog data of all the patients S's electrodes




