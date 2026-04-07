import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os
from pathlib import Path
import argparse
from tqdm import tqdm

# Ensure the parent directory is in the path for package discovery
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from model.mtl_crnn import ScalogramV2Model
from model.losses import FocalLoss, VonMisesLoss
from data.dataloader import get_dataloaders
from utils.checkpoint_utils import ModelCheckpoint

def train(args):
    """
    Main training routine for ScalogramV2.
    
    Implements a multi-task training loop balancing detection, magnitude, 
    and azimuth tasks with specialized losses.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing training on: {device}")
    
    # Model Setup
    model = ScalogramV2Model(pretrained=True).to(device)
    
    # Data Loading
    train_loader, val_loader = get_dataloaders(args.data_path, batch_size=args.batch_size)
    print(f"Dataset loaded: {len(train_loader.dataset)} training samples.")
    
    # Loss Optimization Strategy
    criterion_b = FocalLoss(alpha=1.0, gamma=2.0) # For Imbalanced Detection
    criterion_m = torch.nn.CrossEntropyLoss()      # For Magnitude Grading
    criterion_a = VonMisesLoss(kappa=5.0)          # For Circular Regression
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Dynamic Learning Rate Scheduling
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.lr*10, 
        steps_per_epoch=len(train_loader), 
        epochs=args.epochs
    )
    
    checkpoint_handler = ModelCheckpoint(args.checkpoint_path, monitor='train_loss', mode='min')
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")
        
        for x, y, _, mag, azm in pbar:
            x, y, mag, azm = x.to(device), y.to(device), mag.to(device), azm.to(device)
            
            optimizer.zero_grad()
            out_b, out_m, out_a = model(x)
            
            # Loss Computation (Task-Weighted)
            loss_b = criterion_b(out_b, y)
            loss_m = criterion_m(out_m, mag)
            loss_a = criterion_a(out_a, azm)
            
            # Combined Loss with priority on Detection (Stage 1)
            total_loss = loss_b + 0.1 * loss_m + 0.05 * loss_a
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += total_loss.item()
            pbar.set_postfix({'TotalLoss': f"{total_loss.item():.4f}", 'LR': f"{scheduler.get_last_lr()[0]:.1e}"})
            
        avg_loss = epoch_loss / len(train_loader)
        print(f" -> Epoch Summary: Avg Loss = {avg_loss:.5f}")
        checkpoint_handler(avg_loss, model, optimizer, epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ScalogramV2 Research Training Entry Point")
    parser.add_argument('--data_path', type=str, default='data/sample_data.h5', help="Path to HDF5 tensor dataset")
    parser.add_argument('--checkpoint_path', type=str, default='models/best_model.pth', help="Path to save best weights")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training cycles")
    parser.add_argument('--batch_size', type=int, default=32, help="Samples per batch")
    parser.add_argument('--lr', type=float, default=1e-4, help="Initial learning rate")
    args = parser.parse_args()
    
    train(args)
