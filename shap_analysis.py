#!/usr/bin/env python3
"""
Standalone script for SHAP analysis of assembly bias prediction.
Performs K-fold cross-validation with SHAP feature importance analysis.
"""

import numpy as np
from astropy.table import Table
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor
from scipy.stats import rankdata, spearmanr
import shap
import json
import sys
import joblib


def main():
    # ========================================================================
    # 1. Imports and setup
    # ========================================================================
    flamingo_path = "flamingo-data/l1_hydro_m8_dmo_m10.hdf5"
    delta_b_path = "delta_b.npy"
    
    # ========================================================================
    # 2. Load data (centrals only)
    # ========================================================================
    print("Loading data from HDF5 file...")
    gals = Table.read(flamingo_path, path="galaxies")
    halos = Table.read(flamingo_path, path="halos")
    
    # Select centrals
    is_central = (gals['type'] == b'central')
    gals_c = gals[is_central]
    
    # Extract columns (excluding luminosity bands)
    mstar = np.asarray(gals_c['mstar'], dtype=float)
    sfr = np.asarray(gals_c['sfr'], dtype=float)
    f_disk = np.asarray(gals_c['f_disk'], dtype=float)
    rhalf = np.asarray(gals_c['rhalf'], dtype=float)
    metallicity = np.asarray(gals_c['metallicity'], dtype=float)
    msmbh = np.asarray(gals_c['msmbh'], dtype=float)
    age = np.asarray(gals_c['age'], dtype=float)
    halo_index = np.asarray(gals_c['halo_index'], dtype=np.int64)
    
    # Apply validity mask
    valid = (
        (halo_index >= 0) & (halo_index < len(halos)) &
        np.isfinite(mstar) & (mstar > 0) &
        np.isfinite(sfr) & (sfr > 0) &
        np.isfinite(f_disk) &
        np.isfinite(rhalf) &
        np.isfinite(metallicity) &
        np.isfinite(msmbh) &
        np.isfinite(age)
    )
    
    mstar = mstar[valid]
    sfr = sfr[valid]
    f_disk = f_disk[valid]
    rhalf = rhalf[valid]
    metallicity = metallicity[valid]
    msmbh = msmbh[valid]
    age = age[valid]
    halo_index = halo_index[valid]
    
    # Build logM
    logM = np.log10(mstar)
    
    print(f"Loaded {len(logM)} central galaxies after filtering")
    
    # ========================================================================
    # 3. Bring in the target
    # ========================================================================
    print(f"Loading delta_b from {delta_b_path}...")
    if not Path(delta_b_path).exists():
        print(f"ERROR: {delta_b_path} not found. Please ensure delta_b.npy exists.")
        sys.exit(1)
    
    delta_b = np.load(delta_b_path)
    
    if len(delta_b) != len(halos):
        print(f"ERROR: delta_b length ({len(delta_b)}) does not match halos length ({len(halos)})")
        sys.exit(1)
    
    # Map centrals to their halo residuals
    y = delta_b[halo_index]
    
    # Drop non-finite y
    mask_y = np.isfinite(y)
    logM = logM[mask_y]
    sfr = sfr[mask_y]
    f_disk = f_disk[mask_y]
    rhalf = rhalf[mask_y]
    metallicity = metallicity[mask_y]
    msmbh = msmbh[mask_y]
    age = age[mask_y]
    y = y[mask_y]
    
    print(f"After filtering non-finite y: {len(y)} samples")
    
    # ========================================================================
    # 4. Mass-bin ranking (0.1 dex)
    # ========================================================================
    print("Performing mass-bin ranking...")
    
    # Make 0.1 dex bins in logM
    bin_edges = np.arange(logM.min(), logM.max() + 0.1, 0.1)
    bin_idx = np.digitize(logM, bin_edges) - 1
    
    # Features to rank (all except mstar)
    features_to_rank = {
        'sfr': sfr,
        'f_disk': f_disk,
        'rhalf': rhalf,
        'metallicity': metallicity,
        'msmbh': msmbh,
        'age': age
    }
    
    # Rank each feature within its mass bin
    ranked_features = {}
    for feat_name, feat_values in features_to_rank.items():
        feat_ranked = np.empty_like(feat_values, dtype=float)
        for b in np.unique(bin_idx):
            sel = (bin_idx == b)
            if not np.any(sel):
                continue
            # Rank with method="average"
            ranks = rankdata(feat_values[sel], method="average")
            # Divide by N_bin to get [0,1]
            N_bin = sel.sum()
            feat_ranked[sel] = ranks / N_bin
        ranked_features[feat_name] = feat_ranked
    
    # Build feature matrix X
    # Order: logM, ranked_sfr, ranked_f_disk, ranked_rhalf, 
    #        ranked_metallicity, ranked_msmbh, ranked_age
    X = np.column_stack([
        logM,  # real value, not ranked
        ranked_features['sfr'],
        ranked_features['f_disk'],
        ranked_features['rhalf'],
        ranked_features['metallicity'],
        ranked_features['msmbh'],
        ranked_features['age']
    ])
    
    feature_names = [
        'logM',
        'sfr_ranked',
        'f_disk_ranked',
        'rhalf_ranked',
        'metallicity_ranked',
        'msmbh_ranked',
        'age_ranked'
    ]
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Feature names: {feature_names}")
    
    # ========================================================================
    # 5. Model definition
    # ========================================================================
    print("Defining model pipeline...")
    
    # Define base MLPRegressor with the same hyperparameters
    base_mlp = MLPRegressor(
        activation="relu",
        alpha=0.001,
        batch_size=1024,
        early_stopping=True,
        hidden_layer_sizes=(128, 64, 32),
        learning_rate="constant",
        learning_rate_init=0.001,
        max_iter=5000,
        n_iter_no_change=20,
        random_state=64,
        solver="adam",
        validation_fraction=0.1,
        shuffle=True,
        verbose=False
    )
    
    # Wrap MLPRegressor in BaggingRegressor
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("bagging", BaggingRegressor(
            estimator=base_mlp,
            n_estimators=10,
            random_state=64,
            n_jobs=-1
        ))
    ])
    
    # ========================================================================
    # 6. K-fold loop (robust SHAP)
    # ========================================================================
    print("Starting K-fold cross-validation with SHAP analysis...")
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=64)
    
    # Storage for per-fold results
    mean_abs_shap_list = []
    spearman_corr_list = []
    shap_vals_list = []  # Store actual SHAP values for each fold
    X_explain_list = []  # Store X_explain for each fold
    shap_explanation_list = []  # Store SHAP Explanation objects for each fold
    
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        print(f"\nFold {fold_idx + 1}/5")
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit model on train split
        print(f"  Training model on {len(X_train)} samples...")
        model.fit(X_train, y_train)
        
        # Choose background for SHAP from train split
        bg_size = min(200, X_train.shape[0])
        rng = np.random.default_rng(64 + fold_idx)
        bg_idx = rng.choice(X_train.shape[0], size=bg_size, replace=False)
        X_bg = X_train[bg_idx]
        
        # Use all test points to explain
        n_explain = X_test.shape[0]
        X_explain = X_test
        
        print(f"  Computing SHAP values for all {n_explain} test samples...")
        print(f"  Using {bg_size} background samples...")
        
        # Build explainer and compute SHAP values
        explainer = shap.KernelExplainer(model.predict, X_bg)
        shap_vals = explainer.shap_values(X_explain)
        
        # Convert to numpy array if needed
        shap_vals = np.array(shap_vals)
        
        # Get base value (expected value) from explainer
        base_value = explainer.expected_value
        
        # Create SHAP Explanation object
        explanation = shap.Explanation(
            values=shap_vals,
            base_values=np.full(len(X_explain), base_value),
            data=X_explain,
            feature_names=feature_names
        )
        
        # Compute mean absolute SHAP per feature
        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)  # shape (n_features,)
        mean_abs_shap_list.append(mean_abs_shap)
        
        # Compute Spearman correlation for each feature
        corr_per_feature = []
        for j in range(shap_vals.shape[1]):
            corr_j, _ = spearmanr(shap_vals[:, j], X_explain[:, j])
            corr_per_feature.append(corr_j)
        
        # Convert to numpy array for consistency (shape: (n_features,))
        corr_per_feature = np.array(corr_per_feature)
        spearman_corr_list.append(corr_per_feature)
        
        # Store SHAP values, X_explain, and Explanation object for later analysis
        shap_vals_list.append(shap_vals)
        X_explain_list.append(X_explain)
        shap_explanation_list.append(explanation)
        
        print(f"  Mean |SHAP| per feature: {mean_abs_shap}")
        print(f"  Spearman correlations: {corr_per_feature}")
    
    # ========================================================================
    # 7. Aggregate over folds
    # ========================================================================
    print("\nAggregating results across folds...")
    
    # Stack per-fold results
    mean_abs_shap_array = np.array(mean_abs_shap_list)  # shape (n_folds, n_features)
    spearman_corr_array = np.array(spearman_corr_list)  # shape (n_folds, n_features)
    
    # Take mean across folds
    mean_abs = np.mean(mean_abs_shap_array, axis=0)  # shape (n_features,)
    corr_mean = np.mean(spearman_corr_array, axis=0)  # shape (n_features,)
    
    # ========================================================================
    # 8. Save results with timestamp
    # ========================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Save SHAP values and X_explain as numpy arrays for easy loading
    # Note: Each fold may have different number of test samples, so we save as a list
    # Convert each to numpy array to ensure consistency
    shap_vals_list_arrays = [np.array(sv) for sv in shap_vals_list]
    X_explain_list_arrays = [np.array(xe) for xe in X_explain_list]
    
    # Get shapes for reporting (may vary per fold)
    shap_shapes = [sv.shape for sv in shap_vals_list_arrays]
    X_explain_shapes = [xe.shape for xe in X_explain_list_arrays]
    
    shap_vals_path = results_dir / f"shap_values_{timestamp}.npy"
    X_explain_path = results_dir / f"X_explain_{timestamp}.npy"
    feature_names_path = results_dir / f"feature_names_{timestamp}.npy"
    shap_explanations_path = results_dir / f"shap_explanations_{timestamp}.bz2"
    
    # Save as list of arrays (using allow_pickle to preserve list structure)
    # Wrap in object array to handle variable-sized arrays per fold
    np.save(shap_vals_path, np.array(shap_vals_list_arrays, dtype=object), allow_pickle=True)
    np.save(X_explain_path, np.array(X_explain_list_arrays, dtype=object), allow_pickle=True)
    np.save(feature_names_path, np.array(feature_names))
    
    # Save SHAP Explanation objects using joblib
    # Note: We save as a list, one Explanation object per fold
    joblib.dump(shap_explanation_list, filename=str(shap_explanations_path), compress=('bz2', 9))
    
    print(f"  Saved SHAP values to: {shap_vals_path}")
    print(f"  Saved X_explain to: {X_explain_path}")
    print(f"  Saved feature names to: {feature_names_path}")
    print(f"  Saved SHAP Explanation objects to: {shap_explanations_path}")
    
    # Build output dict
    out = {
        "timestamp": datetime.now().isoformat(),
        "feature_names": feature_names,
        "mean_abs_shap_per_feature": mean_abs.tolist(),
        "spearman_corr_per_feature": corr_mean.tolist(),
        "n_folds": 5,
        "background_size": bg_size,
        "shap_values_path": str(shap_vals_path),
        "X_explain_path": str(X_explain_path),
        "feature_names_path": str(feature_names_path),
        "shap_explanations_path": str(shap_explanations_path),
        "shap_values_shapes_per_fold": [list(s) for s in shap_shapes],
        "X_explain_shapes_per_fold": [list(s) for s in X_explain_shapes],
        "n_explain_per_fold": [int(s[0]) for s in shap_shapes]
    }
    
    # Save to JSON
    output_path = results_dir / f"results_shap_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(out, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # ========================================================================
    # 9. Print a short summary
    # ========================================================================
    print("\n" + "="*60)
    print("SHAP Analysis Summary")
    print("="*60)
    
    # Get top 5 features by mean |SHAP|
    top_idx = np.argsort(mean_abs)[::-1]
    
    print("\nTop features by mean |SHAP|:")
    print("-" * 60)
    for i, idx in enumerate(top_idx, 1):
        sign = "+" if corr_mean[idx] >= 0 else "-"
        print(f"{i}. {feature_names[idx]:20s}  "
              f"mean|SHAP|={mean_abs[idx]:.6f}  "
              f"Spearman={sign}{abs(corr_mean[idx]):.4f}")
    
    print("\n" + "="*60)
    print(f"Full results saved to: {output_path}")
    print(f"SHAP values saved to: {shap_vals_path}")
    print(f"X_explain saved to: {X_explain_path}")
    print(f"Feature names saved to: {feature_names_path}")
    print(f"SHAP Explanation objects saved to: {shap_explanations_path}")
    print("\nShapes per fold:")
    for i, (s_shape, x_shape) in enumerate(zip(shap_shapes, X_explain_shapes), 1):
        print(f"  Fold {i}: SHAP {s_shape}, X_explain {x_shape}")
    print("\nTo load the data for analysis:")
    print(f"  shap_vals_list = np.load('{shap_vals_path}', allow_pickle=True)  # list of arrays, one per fold")
    print(f"  X_explain_list = np.load('{X_explain_path}', allow_pickle=True)  # list of arrays, one per fold")
    print(f"  feature_names = np.load('{feature_names_path}', allow_pickle=True)")
    print(f"  shap_explanations_list = joblib.load('{shap_explanations_path}')  # list of Explanation objects, one per fold")
    print(f"  # Access fold i: shap_vals_list[i] has shape {shap_shapes[0] if shap_shapes else 'varies'}")
    print(f"  # Access fold i: shap_explanations_list[i] is a SHAP Explanation object")
    print("="*60)


if __name__ == "__main__":
    main()

