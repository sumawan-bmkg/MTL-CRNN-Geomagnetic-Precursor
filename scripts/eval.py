import torch
import sys
from pathlib import Path
import argparse
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from model.mtl_crnn import ScalogramV2Model
from data.dataloader import GeomagneticHDF5Dataset
from torch.utils.data import DataLoader

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    model = ScalogramV2Model(pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Dataset
    dataset = GeomagneticHDF5Dataset(args.data_path, group_name='val')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    all_preds = []
    all_targets = []
    
    print(f"Evaluating on {len(dataset)} samples...")
    with torch.no_grad():
        for x, y, _, _, _ in loader:
            x = x.to(device)
            out_b, _, _ = model(x)
            preds = torch.argmax(out_b, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.numpy())
            
    print("\n--- Detection Performance (Stage 1) ---")
    print(classification_report(all_targets, all_preds, target_names=['Quiet', 'Precursor']))
    
    cm = confusion_matrix(all_targets, all_preds)
    print("Confusion Matrix:")
    print(cm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='data/sample_data.h5')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    evaluate(args)
