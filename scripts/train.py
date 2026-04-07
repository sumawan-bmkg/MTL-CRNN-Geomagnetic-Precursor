import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os
from pathlib import Path
import argparse
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from model.mtl_crnn import ScalogramV2Model
from model.losses import FocalLoss, VonMisesLoss, GeomagneticPenaltyLoss
from data.dataloader import get_dataloaders
from utils.checkpoint_utils import ModelCheckpoint

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize Model
    model = ScalogramV2Model(pretrained=True).to(device)
    
    # Dataloaders
    train_loader, val_loader = get_dataloaders(args.data_path, batch_size=args.batch_size)
    
    # Losses & Optimizer
    criterion_b = FocalLoss()
    criterion_m = torch.nn.CrossEntropyLoss()
    criterion_a = VonMisesLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr*10, 
                                            steps_per_epoch=len(train_loader), 
                                            epochs=args.epochs)
    
    checkpoint = ModelCheckpoint(args.checkpoint_path)
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for x, y, is_storm, mag, azm in pbar:
            x, y, is_storm, mag, azm = x.to(device), y.to(device), is_storm.to(device), mag.to(device), azm.to(device)
            
            optimizer.zero_grad()
            out_b, out_m, out_a = model(x)
            
            # Multi-Task Loss Weighting
            loss_b = criterion_b(out_b, y)
            loss_m = criterion_m(out_m, mag)
            loss_a = criterion_a(out_a, azm)
            
            total_loss = loss_b + 0.1 * loss_m + 0.05 * loss_a
            
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += total_loss.item()
            pbar.set_postfix({'loss': f"{total_loss.item():.4f}"})
            
        # Basic Validation Logic (Summary)
        print(f"Epoch {epoch+1} Completed. Avg Loss: {train_loss/len(train_loader):.4f}")
        checkpoint(train_loss/len(train_loader), model, optimizer, epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/sample_data.h5')
    parser.add_argument('--checkpoint_path', type=str, default='models/best_model.pth')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    train(args)
