import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np

class GeomagneticHDF5Dataset(Dataset):
    """
    High-performance HDF5 Dataset for Geomagnetic Tensors.
    Reads directly from .h5 to avoid I/O overhead of individual files.
    """
    def __init__(self, h5_file_path, group_name='train', transform=None):
        self.h5_file_path = str(h5_file_path)
        self.group_name = group_name
        self.transform = transform
        
        # Pre-calculate length
        with h5py.File(self.h5_file_path, 'r') as hf:
            if self.group_name not in hf:
                raise KeyError(f"Group {self.group_name} not found in {h5_file_path}")
            self.length = hf[self.group_name]['tensors'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Open HDF5 locally in worker for thread-safety
        with h5py.File(self.h5_file_path, 'r') as hf:
            grp = hf[self.group_name]
            x_tensor = grp['tensors'][idx]
            y = grp['label_event'][idx]
            is_storm = grp['is_storm'][idx]
            mag = grp['label_mag'][idx]
            azm = grp['label_azm'][idx]
            
        x_tensor = torch.from_numpy(x_tensor).float()
        
        if self.transform:
            x_tensor = self.transform(x_tensor)
            
        return x_tensor, y, is_storm, mag, azm

def get_dataloaders(h5_path, batch_size=32, num_workers=4):
    """
    Convenience function to create train/val loaders.
    """
    train_ds = GeomagneticHDF5Dataset(h5_path, 'train')
    val_ds = GeomagneticHDF5Dataset(h5_path, 'val')
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader
