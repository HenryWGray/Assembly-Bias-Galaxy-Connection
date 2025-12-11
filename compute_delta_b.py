#!/usr/bin/env python3
"""
Standalone script to compute delta_b (residual bias) values for halos.
Follows the structure from perpart-perhalo.ipynb to compute assembly bias residuals.
"""

import numpy as np
from astropy.table import Table
from scipy.spatial import cKDTree
from datetime import datetime
import sys


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
# Helper Functions
# ============================================================================

def shell_counts(points, tree_all, rmin, rmax, batch=50000):
    """
    Count particles in shell around each point.
    
    Parameters
    ----------
    points : array-like
        Points to count around
    tree_all : cKDTree
        KD tree of all particles
    rmin : float
        Inner radius of shell
    rmax : float
        Outer radius of shell
    batch : int
        Batch size for processing
    
    Returns
    -------
    counts : ndarray
        Count of particles in shell around each point
    """
    counts = np.empty(len(points), dtype=np.int32)
    for i in range(0, len(points), batch):
        sub = points[i:i+batch]
        # neighbors within rmax and rmin
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


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    # File paths
    flamingo_path = "flamingo-data/l1_hydro_m8_dmo_m10.hdf5"
    output_path = "delta_b.npy"
    
    print("="*60)
    print("Computing delta_b (residual bias) for halos")
    print("="*60)
    print(f"Input file: {flamingo_path}")
    print(f"Output file: {output_path}")
    print(f"Shell radii: R_MIN={R_MIN}, R_MAX={R_MAX}")
    print(f"Polynomial degree: {POLY_DEG}")
    print()
    
    # ========================================================================
    # 1. Load data
    # ========================================================================
    print("Step 1: Loading data from HDF5 file...")
    try:
        halos_tab = Table.read(flamingo_path, path="halos")
        parts_tab = Table.read(flamingo_path, path="particles")
    except Exception as e:
        print(f"ERROR: Failed to load data from {flamingo_path}")
        print(f"Error: {e}")
        sys.exit(1)
    
    particle_pos = np.array(parts_tab["pos"], dtype=np.float64)
    halo_pos = np.array(halos_tab["pos"], dtype=np.float64)
    halo_mass = np.array(halos_tab["mtot"], dtype=np.float64)
    
    print(f"  Loaded {len(particle_pos)} particles")
    print(f"  Loaded {len(halo_pos)} halos")
    print()
    
    # ========================================================================
    # 2. Fit periodic box
    # ========================================================================
    print("Step 2: Fitting data to periodic box...")
    pmin = np.minimum(particle_pos.min(axis=0), halo_pos.min(axis=0))
    pmax = np.maximum(particle_pos.max(axis=0), halo_pos.max(axis=0))
    Lvec = pmax - pmin
    
    # Strict upper bound
    Lvec = np.nextafter(Lvec.astype(np.float64), np.float64(np.inf))
    
    # Wrap into [0, Lvec)
    pp = (particle_pos - pmin) % Lvec
    hp = (halo_pos - pmin) % Lvec
    
    print(f"  Box size: {Lvec}")
    print()
    
    # ========================================================================
    # 3. Build KD trees
    # ========================================================================
    print("Step 3: Building KD trees...")
    tree_all = cKDTree(pp, boxsize=Lvec)
    tree_halo = cKDTree(hp, boxsize=Lvec)
    print("  KD trees built")
    print()
    
    # ========================================================================
    # 4. Compute baseline (random particles)
    # ========================================================================
    print("Step 4: Computing random baseline from particles...")
    rng = np.random.default_rng(RNG_SEED)
    
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
    
    print(f"  Particle baseline: mean={mean_pp:.3f}, SEM={sem_pp:.3f}, "
          f"rel_err={100*rel_err_pp:.2f}%")
    print()
    
    # ========================================================================
    # 5. Compute uniform random baseline
    # ========================================================================
    print("Step 5: Computing uniform random baseline...")
    
    def sample_uniform(n):
        u = rng.random((n, 3))
        return u * Lvec
    
    def batch_mean_shell_uniform(n):
        pts = sample_uniform(n)
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
    
    print(f"  Uniform baseline: mean={mean_uniform:.3f}, SEM={sem_uniform:.3f}, "
          f"rel_err={100*rel_err_uniform:.2f}%")
    print()
    
    # ========================================================================
    # 6. Compute shell counts around halos
    # ========================================================================
    print("Step 6: Computing shell counts around halos...")
    print(f"  Processing in batches of {HALO_BATCH}...")
    halo_counts = shell_counts(hp, tree_all, R_MIN, R_MAX, batch=HALO_BATCH)
    
    mean_ph = halo_counts.mean()
    print(f"  Halo mean particles per halo: {mean_ph:.3f}")
    print()
    
    # ========================================================================
    # 7. Compute bias per individual halo
    # ========================================================================
    print("Step 7: Computing bias per individual halo...")
    b_i = (halo_counts - mean_uniform) / (mean_pp - mean_uniform)
    
    bias_all = (mean_ph - mean_uniform) / (mean_pp - mean_uniform)
    print(f"  Overall bias: {bias_all:.3f}")
    print(f"  Bias stats: mean={np.nanmean(b_i):.4f}, std={np.nanstd(b_i):.4f}")
    print()
    
    # ========================================================================
    # 8. Fit polynomial to remove mass trend
    # ========================================================================
    print(f"Step 8: Fitting polynomial (degree {POLY_DEG}) to remove mass trend...")
    logM = np.log10(halo_mass)
    coef = np.polyfit(logM, b_i, deg=POLY_DEG)
    p = np.poly1d(coef)
    
    print(f"  Polynomial coefficients: {coef}")
    print()
    
    # ========================================================================
    # 9. Compute delta_b (residual bias)
    # ========================================================================
    print("Step 9: Computing delta_b (residual bias)...")
    delta_b = b_i - p(logM)
    
    print(f"  Delta_b stats:")
    print(f"    Mean: {np.nanmean(delta_b):.6f}")
    print(f"    Std:  {np.nanstd(delta_b):.6f}")
    print(f"    Min:  {np.nanmin(delta_b):.6f}")
    print(f"    Max:  {np.nanmax(delta_b):.6f}")
    print()
    
    # ========================================================================
    # 10. Save to file
    # ========================================================================
    print(f"Step 10: Saving delta_b to {output_path}...")
    np.save(output_path, delta_b)
    print(f"  Saved {len(delta_b)} delta_b values")
    print()
    
    print("="*60)
    print("SUCCESS: delta_b computation completed!")
    print(f"Output saved to: {output_path}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*60)


if __name__ == "__main__":
    main()

