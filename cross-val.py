#!/usr/bin/env python3
"""
Cross-validation script for assembly bias prediction in dark matter halos.
Transforms the pipeline from perpart-perhalo.ipynb into a cluster-executable script.
"""

import numpy as np
from astropy.table import Table
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import argparse
import sys
import os
from datetime import datetime
import json

# ============================================================================
# Configuration
# ============================================================================

# Shell radii for bias calculation
R_MIN, R_MAX = 6.0, 9.0

# Random baseline parameters
N_RAND = 100000
BATCH = 10000
RNG_SEED = 64

# Halo batch size for shell counting
HALO_BATCH = 50000

# Polynomial fit degree for mass-bias relationship
POLY_DEG = 4

# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

def load_data(path):
    """Load halo and particle data from HDF5 file."""
    print(f"Loading data from {path}...")
    halos_tab = Table.read(path, path="halos")
    parts_tab = Table.read(path, path="particles")
    
    particle_pos = np.array(parts_tab["pos"], dtype=np.float64)
    halo_pos = np.array(halos_tab["pos"], dtype=np.float64)
    halo_mass = np.array(halos_tab["mtot"], dtype=np.float64)
    
    print(f"Loaded {len(particle_pos)} particles and {len(halo_pos)} halos")
    return particle_pos, halo_pos, halo_mass

def fit_periodic_box(particle_pos, halo_pos):
    """Fit data to a periodic box and wrap coordinates."""
    print("Fitting data to periodic box...")
    pmin = np.minimum(particle_pos.min(axis=0), halo_pos.min(axis=0))
    pmax = np.maximum(particle_pos.max(axis=0), halo_pos.max(axis=0))
    Lvec = pmax - pmin
    
    # Strict upper bound
    Lvec = np.nextafter(Lvec.astype(np.float64), np.float64(np.inf))
    
    # Wrap into [0, Lvec)
    pp = (particle_pos - pmin) % Lvec
    hp = (halo_pos - pmin) % Lvec
    
    print(f"Box size: {Lvec}")
    return pp, hp, Lvec

# ============================================================================
# Bias Calculation
# ============================================================================

def compute_baseline(tree_all, pp, Lvec, rng):
    """Compute random baselines for bias calculation."""
    print("Computing random baseline from particles...")
    
    # Baseline 1: Random particles from dataset
    def batch_mean_shell(batch_pts):
        tb = cKDTree(batch_pts, boxsize=Lvec)
        cmax = tb.count_neighbors(tree_all, R_MAX)
        cmin = tb.count_neighbors(tree_all, R_MIN)
        return (cmax - cmin) / len(batch_pts)
    
    means_pp = []
    n_done = 0
    while n_done < N_RAND:
        n_take = min(BATCH, N_RAND - n_done)
        idx = rng.integers(0, len(pp), size=n_take)
        means_pp.append(batch_mean_shell(pp[idx]))
        n_done += n_take
    
    means_pp = np.array(means_pp)
    mean_pp = means_pp.mean()
    sem_pp = means_pp.std(ddof=1) / np.sqrt(len(means_pp))
    rel_err_pp = sem_pp / mean_pp
    
    print(f"Particle baseline: mean={mean_pp:.3f}, SEM={sem_pp:.3f}, rel_err={100*rel_err_pp:.2f}%")
    
    # Baseline 2: Uniform random points in box
    print("Computing uniform random baseline...")
    
    def batch_mean_shell_uniform(n):
        u = rng.random((n, 3))
        pts = u * Lvec
        tb = cKDTree(pts, boxsize=Lvec)
        cmax = tb.count_neighbors(tree_all, R_MAX)
        cmin = tb.count_neighbors(tree_all, R_MIN)
        return (cmax - cmin) / n
    
    means_uniform = []
    n_done = 0
    while n_done < N_RAND:
        n_take = min(BATCH, N_RAND - n_done)
        means_uniform.append(batch_mean_shell_uniform(n_take))
        n_done += n_take
    
    means_uniform = np.array(means_uniform)
    mean_uniform = means_uniform.mean()
    sem_uniform = means_uniform.std(ddof=1) / np.sqrt(len(means_uniform))
    rel_err_uniform = sem_uniform / mean_uniform
    
    print(f"Uniform baseline: mean={mean_uniform:.3f}, SEM={sem_uniform:.3f}, rel_err={100*rel_err_uniform:.2f}%")
    
    return mean_uniform, mean_pp

def shell_counts(points, tree_all, rmin, rmax, batch=50000):
    """Count particles in shell around each point."""
    counts = np.empty(len(points), dtype=np.int32)
    for i in range(0, len(points), batch):
        sub = points[i:i+batch]
        idx_rmax = tree_all.query_ball_point(sub, r=rmax, workers=-1)
        if rmin > 0:
            idx_rmin = tree_all.query_ball_point(sub, r=rmin, workers=-1)
            counts[i:i+batch] = np.fromiter(
                (len(a) - len(b) for a, b in zip(idx_rmax, idx_rmin)),
                dtype=np.int32, count=len(sub)
            )
        else:
            counts[i:i+batch] = np.fromiter(
                (len(a) for a in idx_rmax),
                dtype=np.int32, count=len(sub)
            )
    return counts

def compute_halo_bias(hp, tree_all, tree_halo, mean_uniform, mean_pp, Lvec):
    """Compute bias for each halo."""
    print("Computing shell counts around halos...")
    halo_counts = shell_counts(hp, tree_all, R_MIN, R_MAX, batch=HALO_BATCH)
    
    # Calculate bias per individual halo
    mean_ph = halo_counts.mean()
    b_i = (halo_counts - mean_uniform) / (mean_pp - mean_uniform)
    
    print(f"Halo mean particles per halo: {mean_ph:.3f}")
    print(f"Overall bias: {(mean_ph - mean_uniform) / (mean_pp - mean_uniform):.3f}")
    
    return b_i, halo_counts

def compute_delta_b(b_i, halo_mass, deg=POLY_DEG):
    """Compute residual bias (delta_b) after removing mass trend."""
    print(f"Computing delta_b using polynomial fit (degree {deg})...")
    logM = np.log10(halo_mass)
    coef = np.polyfit(logM, b_i, deg=deg)
    p = np.poly1d(coef)
    
    delta_b = b_i - p(logM)
    
    print(f"Delta_b stats: mean={np.nanmean(delta_b):.4f}, std={np.nanstd(delta_b):.4f}")
    return delta_b

# ============================================================================
# Feature Preparation
# ============================================================================

def prepare_features(path, delta_b):
    """Prepare galaxy features for neural network training."""
    print("Loading galaxy data and preparing features...")
    
    gals = Table.read(path, path='galaxies')
    halos = Table.read(path, path='halos')
    
    # Select centrals
    is_central = (gals['type'] == b'central')
    gals_c = gals[is_central]
    
    # Pull columns
    lum = np.asarray(gals_c["luminosity"], float)
    
    feat_dict = {
        "mstar":      np.asarray(gals_c["mstar"], float),
        "sfr":        np.asarray(gals_c["sfr"], float),
        "f_disk":     np.asarray(gals_c["f_disk"], float),
        "rhalf":      np.asarray(gals_c["rhalf"], float),
        "lum0":       lum[:, 0],
        "lum1":       lum[:, 1],
        "lum2":       lum[:, 2],
        "lum3":       lum[:, 3],
        "lum4":       lum[:, 4],
        "lum5":       lum[:, 5],
        "lum6":       lum[:, 6],
        "lum7":       lum[:, 7],
        "lum8":       lum[:, 8],
        "metallicity": np.asarray(gals_c["metallicity"], float),
        "msmbh":       np.asarray(gals_c["msmbh"], float),
        "age":         np.asarray(gals_c["age"], float),
    }
    hidx = np.asarray(gals_c["halo_index"], np.int64)
    
    # Validity mask
    valid = (
        (hidx >= 0) & (hidx < len(halos)) &
        np.isfinite(feat_dict["mstar"]) & (feat_dict["mstar"] > 0) &
        np.isfinite(feat_dict["sfr"]) & (feat_dict["sfr"] > 0)
    )
    
    for k in feat_dict:
        feat_dict[k] = feat_dict[k][valid]
    hidx = hidx[valid]
    
    # Target (delta_b is per-halo)
    y = np.asarray(delta_b, dtype=float)[hidx]
    mask_y = np.isfinite(y)
    for k in feat_dict:
        feat_dict[k] = feat_dict[k][mask_y]
    y = y[mask_y]
    
    # Build logMstar and mass bins
    logM = np.log10(feat_dict["mstar"])
    bin_edges = np.arange(logM.min(), logM.max() + 0.1, 0.1)
    bin_idx = np.digitize(logM, bin_edges) - 1
    
    # Replace features (except mstar) by quantile within mass bin
    features_to_rank = [k for k in feat_dict.keys() if k != "mstar"]
    
    print("Ranking features within mass bins...")
    for k in features_to_rank:
        arr = feat_dict[k]
        arr_rank = np.empty_like(arr, dtype=float)
        for b in np.unique(bin_idx):
            sel = (bin_idx == b)
            if not np.any(sel):
                continue
            ranks = rankdata(arr[sel], method='average')
            arr_rank[sel] = ranks / sel.sum()
        feat_dict[k] = arr_rank
    
    # Construct X: first column = logM, rest = ranked features
    X_cols = [logM] + [feat_dict[k] for k in features_to_rank]
    X = np.column_stack(X_cols)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y

# ============================================================================
# Model Training
# ============================================================================

def train_model(X, y, param_grid=None, n_jobs=1, cv_splits=5, random_state=64):
    """Train neural network with cross-validation."""
    print("Training neural network with cross-validation...")
    
    if param_grid is None:
        param_grid = {
            'mlp__hidden_layer_sizes': [(32,), (64,), (128,), (128,64), (128,64,32)],
            'mlp__activation': ['relu', 'tanh'],
            'mlp__solver': ['adam', 'lbfgs'],
            'mlp__alpha': [1e-5, 1e-4, 1e-3, 1e-2],
            'mlp__learning_rate_init': [1e-3, 3e-4, 1e-4],
            "mlp__batch_size": [256, 1024],
        }
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            max_iter=5000,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=random_state
        ))
    ])
    
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    gs = GridSearchCV(
        pipe, param_grid, cv=cv, scoring="r2", 
        n_jobs=n_jobs, verbose=1
    )
    
    gs.fit(X_train, y_train)
    
    print(f"\nBest CV R2: {gs.best_score_:.6f}")
    print(f"Best params: {gs.best_params_}")
    
    y_pred = gs.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Test R2: {test_r2:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    
    return gs.best_estimator_, gs.best_score_, test_r2, test_mae, gs.best_params_

# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train neural network to predict assembly bias in dark matter halos'
    )
    parser.add_argument(
        '--data-path', type=str, 
        default='flamingo-data/l1_hydro_m8_dmo_m10.hdf5',
        help='Path to HDF5 data file'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory (default: creates timestamped directory)'
    )
    parser.add_argument(
        '--output-model', type=str, default=None,
        help='Output model filename (default: mlp_ranked.pkl in output directory)'
    )
    parser.add_argument(
        '--n-jobs', type=int, default=1,
        help='Number of parallel jobs for GridSearchCV'
    )
    parser.add_argument(
        '--cv-splits', type=int, default=5,
        help='Number of CV folds'
    )
    parser.add_argument(
        '--random-state', type=int, default=64,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    rng = np.random.default_rng(args.random_state)
    np.random.seed(args.random_state)
    
    # Step 1: Load data
    particle_pos, halo_pos, halo_mass = load_data(args.data_path)
    
    # Step 2: Fit periodic box
    pp, hp, Lvec = fit_periodic_box(particle_pos, halo_pos)
    
    # Step 3: Build KD trees
    print("Building KD trees...")
    tree_all = cKDTree(pp, boxsize=Lvec)
    tree_halo = cKDTree(hp, boxsize=Lvec)
    
    # Step 4: Compute baseline
    mean_uniform, mean_pp = compute_baseline(tree_all, pp, Lvec, rng)
    
    # Step 5: Compute halo bias
    b_i, halo_counts = compute_halo_bias(hp, tree_all, tree_halo, mean_uniform, mean_pp, Lvec)
    
    # Step 6: Compute delta_b
    delta_b = compute_delta_b(b_i, halo_mass)
    
    # Step 7: Prepare features
    X, y = prepare_features(args.data_path, delta_b)
    
    # Step 8: Train model
    best_model, cv_r2, test_r2, test_mae, best_params = train_model(
        X, y, n_jobs=args.n_jobs, 
        cv_splits=args.cv_splits, 
        random_state=args.random_state
    )
    
    # Step 9: Create output directory with timestamp
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results_{timestamp}"
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nCreated output directory: {output_dir}")
    
    # Step 10: Save model
    if args.output_model is None:
        model_path = os.path.join(output_dir, "mlp_ranked.pkl")
    else:
        model_path = os.path.join(output_dir, args.output_model)
    
    print(f"Saving model to {model_path}...")
    joblib.dump(best_model, model_path)
    
    # Step 11: Save metrics
    metrics = {
        "cv_r2": float(cv_r2),
        "test_r2": float(test_r2),
        "test_mae": float(test_mae),
        "timestamp": datetime.now().isoformat(),
        "data_path": args.data_path,
        "n_jobs": args.n_jobs,
        "cv_splits": args.cv_splits,
        "random_state": args.random_state,
        "model_path": model_path,
        "best_params": {k: str(v) for k, v in best_params.items()}
    }
    
    # Also get full MLP parameters from the trained model
    if hasattr(best_model, 'named_steps'):
        mlp_params = best_model.named_steps['mlp'].get_params()
        metrics["mlp_params"] = {k: str(v) for k, v in mlp_params.items()}
    
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Also save a human-readable summary
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Assembly Bias Model Training Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Timestamp: {metrics['timestamp']}\n")
        f.write(f"Data path: {metrics['data_path']}\n")
        f.write(f"Random state: {metrics['random_state']}\n")
        f.write(f"CV splits: {metrics['cv_splits']}\n")
        f.write(f"Parallel jobs: {metrics['n_jobs']}\n\n")
        f.write("-"*60 + "\n")
        f.write("Performance Metrics\n")
        f.write("-"*60 + "\n")
        f.write(f"Cross-validation R²: {cv_r2:.6f}\n")
        f.write(f"Test R²: {test_r2:.6f}\n")
        f.write(f"Test MAE: {test_mae:.6f}\n\n")
        f.write("-"*60 + "\n")
        f.write("Model Information\n")
        f.write("-"*60 + "\n")
        f.write(f"Model saved to: {model_path}\n")
        f.write("\nBest Hyperparameters (from GridSearchCV):\n")
        for k, v in metrics["best_params"].items():
            f.write(f"  {k}: {v}\n")
        if "mlp_params" in metrics:
            f.write("\nFull MLP Parameters:\n")
            for k, v in metrics["mlp_params"].items():
                f.write(f"  {k}: {v}\n")
        f.write("\n" + "="*60 + "\n")
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"\nCV R²: {cv_r2:.6f}")
    print(f"Test R²: {test_r2:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    print("="*60)

if __name__ == "__main__":
    main()
