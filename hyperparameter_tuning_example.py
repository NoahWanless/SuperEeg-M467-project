"""
Hyperparameter Tuning Example
==============================
This script shows how to use the hyperparameter tuning functions.
Make sure you have all your data loaded before running this.
"""

from noah_production_funcs import (
    hyperparameter_tuning, 
    visualize_hyperparameter_results,
    evaluate_predictions,
    test_single_hyperparameter
)
import numpy as np

# ============================================================================
# EXAMPLE USAGE (fill in with your actual data)
# ============================================================================


# Step 1: Load and prepare your data
# (Use code from the commented examples in noah_production_funcs.py)

from pathlib import Path
from tara_preprocessing import (
    preprocessing, 
    get_just_ecog_data, 
    get_electrode_normalized_loc,
    make_patient_correlation_matrix
)

# Load data
data_root = Path("C:\\Users\\owenw\\Code\\School\\data_science_projects\\faces_basic\\data")
registered_dir = Path("registered_outputs")
ecogs = get_just_ecog_data(registered_dir, data_root)
xyz = get_electrode_normalized_loc(registered_dir)
xyz_clean, cleaned = preprocessing(ecogs, xyz, notch_size=2)
patient_corr_mat = make_patient_correlation_matrix(xyz_clean, cleaned)
correlation_matrix = np.load("correlation_matrix.npy")

# ============================================================================
# OPTIONAL: Debug single hyperparameter before running full grid search
# ============================================================================
# IMPORTANT: If you get "Encountered all NA values" error, uncomment the 
# following to test just ONE hyperparameter combination first to see what's wrong:

# test_single_hyperparameter(
#     k=5,
#     r=50, 
#     lamb=0.01,
#     patient_corr_mat=patient_corr_mat,
#     xyz_clean=xyz_clean,
#     ecogs=cleaned,
#     patient_idx=0,
#     training_steps=100  # Use fewer steps for quick testing
# )

# Step 2: Define hyperparameter ranges to test
k_range = [5, 10, 15]                          # Number of neighbors (or RBF scale)
r_range = [50, 100, 150]                       # Rank/complexity of U matrix
lamb_range = [0.001, 0.01, 0.1, 1.0]           # Regularization parameter

# Step 3: Run hyperparameter tuning
tuning_results = hyperparameter_tuning(
    k_range=k_range,
    r_range=r_range,
    lamb_range=lamb_range,
    patient_corr_mat=patient_corr_mat,
    xyz_clean=xyz_clean,
    ecogs=cleaned,
    correlation_matrix=correlation_matrix,
    patient_idx=0,                              # Test on patient 0
    training_steps=500,                         # Can increase for better results
    lr=0.01,
    graph='knn',                                # or 'rbf'
    verbose=True
)

# Step 4: Visualize results
fig = visualize_hyperparameter_results(tuning_results, metric='correlation')

# Step 5: Check if tuning was successful
if tuning_results['best_params'] is None:
    print("\n" + "="*70)
    print("TUNING FAILED - All training runs had errors!")
    print("="*70)
    print("See errors printed above for debugging information.")
    print("\nTips:")
    print("1. Uncomment the test_single_hyperparameter() call to debug one combo")
    print("2. Check that patient_idx < number of patients")
    print("3. Verify data types and shapes are correct")
    print("4. Try smaller r values or different k values")
else:
    # Step 5: Access best results
    print("\n=== BEST PARAMETERS ===")
    print(tuning_results['best_params'])
    print("\n=== BEST METRICS ===")
    print(tuning_results['best_metrics'])

    # Step 6: Access the DataFrame with all results for custom analysis
    results_df = tuning_results['results']
    print("\n=== FULL RESULTS ===")
    print(results_df.to_string())

    # Step 7: Get the best model
    best_k = tuning_results['best_params']['k']
    best_r = tuning_results['best_params']['r']
    best_lamb = tuning_results['best_params']['lamb']
    best_U = tuning_results['all_models'][f'k{best_k}_r{best_r}_lamb{best_lamb}']
    print(f"\nBest U matrix shape: {best_U.shape}")



# ============================================================================
# QUICK REFERENCE OF METRICS EXPLAINED
# ============================================================================

print("""
METRIC EXPLANATIONS:
====================

1. CORRELATION (Pearson Correlation Coefficient):
   - Range: -1 to 1
   - Measures linear relationship between prediction and actual
   - 1 = perfect positive correlation
   - 0 = no correlation
   - -1 = perfect negative correlation
   - Higher is better

2. R² SCORE (Coefficient of Determination):
   - Range: 0 to 1 (can be negative for bad models)
   - Proportion of variance explained by the model
   - 1 = model explains all variance (perfect)
   - 0 = model no better than using the mean
   - Higher is better

3. MSE (Mean Squared Error):
   - Average squared difference between predicted and actual
   - Penalizes large errors more
   - Lower is better

4. RMSE (Root Mean Squared Error):
   - Square root of MSE
   - Same units as the data
   - More interpretable than MSE
   - Lower is better

HYPERPARAMETER MEANINGS:
=========================

1. k (Neighbors / RBF Scale):
   - For KNN graph: number of nearest neighbors to connect
   - For RBF graph: scaling factor for the RBF function
   - Affects graph structure and how neighboring electrodes influence each other
   - Smaller: sparser graph, less smoothing
   - Larger: denser graph, more smoothing

2. r (Rank / Complexity):
   - Number of columns in the U matrix
   - Higher r = more complex model, can capture more variation
   - Too high: overfitting
   - Too low: underfitting
   - Typical range: 50-200

3. λ (Regularization Parameter):
   - Controls strength of the Laplacian regularization term
   - Higher λ: stronger smoothing, more spatial coherence
   - Lower λ: more flexibility, less smoothing
   - Prevents overfitting by enforcing spatial smoothness
   - Typical range: 0.001 to 1.0
""")

# ============================================================================
# TIPS FOR TUNING
# ============================================================================

print("""
TUNING TIPS:
============

1. Start broad, then zoom in:
   - First run with coarse grid (few values per parameter)
   - Identify promising region
   - Run finer grid in that region

2. Consider computational cost:
   - Each combination trains a new model
   - Total combos = len(k_range) × len(r_range) × len(lamb_range)
   - Reduce training_steps for initial explorations (100-200)
   - Increase to 500-1000 for final tuning

3. Evaluate on held-out test patient:
   - Use patient_idx to test on different patients
   - Try multiple patients to ensure robustness

4. Monitor the results DataFrame:
   - Check for NaN values (failed trainings)
   - Look at final_loss values for convergence
   - Identify if one parameter dominates

5. Balance metrics:
   - High correlation and R² typically coincide
   - Low RMSE usually means good predictions
   - Default: maximizing correlation is a good choice
""")
