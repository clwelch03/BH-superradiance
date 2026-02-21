import superradiance as sr
import spin_catalog as sc

import sys
import os
import glob
import h5py
import numpy as np
from scipy.stats import beta

import bilby

import json

# --- Set up the catalog -- #

# Find data files
pe_dir = "/pscratch/sd/c/clwelch/BH-superradiance/gwtc3_pe_samples" 
h5_files = glob.glob(os.path.join(pe_dir, "*_cosmo.h5"))

catalog_data = []

# Extract data from files
for filepath in h5_files:
    # Get event name from the filename (e.g., 'GW191109' from 'IGWN-GWTC3p0-v1-GW200322_091133_PEDataRelease_mixed_cosmo.h5')
    event_name = "_".join(os.path.basename(filepath).split('_')[0:2]).replace("IGWN-GWTC3p0-v1-", "")
    
    try:
        with h5py.File(filepath, 'r') as f:
            keys = list(f.keys())
            
            # GWTC-3 files often contain multiple waveform approximants. 
            # We want the combined 'Mixed' samples if available.
            if 'C01:Mixed' in keys:
                posterior = f['C01:Mixed/posterior_samples']
            elif 'C01:IMRPhenomXPHM' in keys:
                posterior = f['C01:IMRPhenomXPHM/posterior_samples']
            else:
                # Fallback to whatever the first waveform is
                posterior = f[f"{keys[0]}/posterior_samples"]
                
            mass_1_source = posterior['mass_1_source'][:]
            spin_magnitude = posterior['a_1'][:]
            
            # Fit Beta distribution for spin magnitude
            a_fit, b_fit, _, _ = beta.fit(spin_magnitude, floc=0, fscale=1)
            
            # Get median mass
            median_mass = np.median(mass_1_source)
            
            # print(f"[{event_name}] Mass: {median_mass:.1f} M_sol | Spin Beta(alpha={a_fit:.2f}, beta={b_fit:.2f})")
            
            catalog_data.append({
                'event': event_name,
                'mass_1_median': median_mass,
                'spin_alpha': a_fit,
                'spin_beta': b_fit
            })
            
    except Exception as e:
        print(f"Skipping {event_name} due to error: {e}")

print(f"\nSuccessfully prepped {len(catalog_data)} events for the superradiance pipeline!")

# Save beta parameters
serializable_catalog = []
for entry in catalog_data:
    serializable_catalog.append({
        'event': str(entry['event']),
        'mass_1_median': float(entry['mass_1_median']),
        'spin_alpha': float(entry['spin_alpha']),
        'spin_beta': float(entry['spin_beta'])
    })

with open('catalog_metadata.json', 'w') as f:
    json.dump(serializable_catalog, f, indent=4)