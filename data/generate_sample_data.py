import torch
import h5py
import numpy as np
import os
from pathlib import Path

def generate_sample_hdf5(out_path, n_samples=10):
    """
    Generates a tiny dummy HDF5 dataset for dry-run testing.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with h5py.File(out_path, 'w') as hf:
        for group_name in ['train', 'val']:
            grp = hf.create_group(group_name)
            
            # Tensors: 3 channels (Z, H, D), 128 freq, 1440 time
            grp.create_dataset('tensors', data=np.random.randn(n_samples, 3, 128, 1440).astype(np.float16))
            
            # Labels
            grp.create_dataset('label_event', data=np.random.randint(0, 2, n_samples))
            grp.create_dataset('is_storm', data=np.random.randint(0, 2, n_samples))
            grp.create_dataset('label_mag', data=np.random.randint(0, 3, n_samples))
            grp.create_dataset('label_azm', data=np.random.uniform(0, 2*np.pi, n_samples).astype(np.float32))
            
    print(f"Sample dataset generated at: {out_path}")

if __name__ == '__main__':
    out_path = Path(__file__).parent / 'sample_data.h5'
    generate_sample_hdf5(out_path)
