import torch
import sys
from pathlib import Path
import argparse
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from torch.utils.data import DataLoader

# Ensure the parent directory is in the path for package discovery
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from model.mtl_crnn import ScalogramV2Model
from data.dataloader import GeomagneticHDF5Dataset

def evaluate(args):
    """
    Evaluation suite for the ScalogramV2 model.
    
    Performs batch-wise inference on a held-out dataset and computes 
    standard metrics (Precision, Recall, F1-Score) and the Confusion Matrix.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting evaluation on: {device}")
    
    # Load Model with strict=False to allow for flexible head loading if needed
    model = ScalogramV2Model(pretrained=False).to(device)
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return
        
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Dataset Preparation
    dataset = GeomagneticHDF5Dataset(args.data_path, group_name='val')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    all_preds = []
    all_targets = []
    
    print(f"Running inference on {len(dataset)} validation samples...")
    with torch.no_grad():
        for x, y, _, _, _ in loader:
            x = x.to(device)
            out_b, _, _ = model(x)
            
            # Extract detection head (Stage 1)
            preds = torch.argmax(out_b, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.numpy())
            
    # Metrics Reporting
    print("\n" + "="*40)
    print("DETECTION PERFORMANCE REPORT (STAGE 1)")
    print("="*40)
    print(classification_report(all_targets, all_preds, target_names=['Quiet', 'Precursor']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))
    print("="*40)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ScalogramV2 Model Evaluation Script")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the .pth model checkpoint")
    parser.add_argument('--data_path', type=str, default='data/sample_data.h5', help="Path to evaluation HDF5 dataset")
    parser.add_argument('--batch_size', type=int, default=32, help="Samples per batch during inference")
    
    import os # Import here for the check inside evaluate
    args = parser.parse_args()
    
    evaluate(args)
