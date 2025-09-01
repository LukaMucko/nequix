#!/usr/bin/env python3
"""
Simple parallel script to compute SCF density matrices for all molecules in the dataset.
Usage: python parallel_density_compute.py
"""
import os
import pickle
import multiprocessing
from functools import partial
import traceback

# Set environment to avoid JAX memory issues
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from data import scf_density_matrix
from load_atoms import load_dataset


def compute_density_safe(args):
    """
    Safely compute density matrix for a single molecule with error handling.
    Returns (index, density_matrix, error_message)
    """
    idx, atoms = args
    try:
        density_matrix = scf_density_matrix(atoms, 'sto-3g')
        return idx, density_matrix, None
    except Exception as e:
        error_msg = f"Error computing density for molecule {idx}: {str(e)}"
        print(error_msg)
        return idx, None, error_msg


def main():
    print("Loading QM7 dataset...")
    ds = load_dataset('QM7', 'data')
    
    print(f"Loaded {len(ds)} molecules")
    
    # Create argument list for parallel processing
    args_list = [(i, atoms) for i, atoms in enumerate(ds)]
    
    # Use all available CPUs
    n_processes = 32
    print(f"Using {n_processes} parallel processes")
    
    print("Computing SCF density matrices in parallel...")
    
    # Process in parallel
    with multiprocessing.Pool(n_processes) as pool:
        results = pool.map(compute_density_safe, args_list)
    
    # Separate successful results from errors
    densities = {}
    errors = {}
    
    for idx, density_matrix, error_msg in results:
        if error_msg is None:
            densities[idx] = density_matrix
        else:
            errors[idx] = error_msg
    
    print(f"Successfully computed {len(densities)} density matrices")
    print(f"Failed to compute {len(errors)} density matrices")
    
    # Save results
    output_data = {
        'densities': densities,
        'errors': errors,
        'metadata': {
            'basis': 'sto-3g',
            'total_molecules': len(ds),
            'successful': len(densities),
            'failed': len(errors)
        }
    }
    
    output_file = 'densities.pkl'
    print(f"Saving results to {output_file}...")
    
    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"Done! Results saved to {output_file}")
    print(f"Summary: {len(densities)}/{len(ds)} successful computations")
    
    if errors:
        print(f"Errors encountered for indices: {list(errors.keys())}")


if __name__ == "__main__":
    main()
