import torch
import os

class ModelCheckpoint:
    """
    Saves the best model weights during training.
    """
    def __init__(self, filepath, monitor='val_f1', mode='max', verbose=True):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best_score = -float('inf') if mode == 'max' else float('inf')
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

    def __call__(self, current_score, model, optimizer, epoch):
        is_best = False
        if (self.mode == 'max' and current_score > self.best_score) or \
           (self.mode == 'min' and current_score < self.best_score):
            is_best = True
            self.best_score = current_score
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_score': self.best_score,
            }
            torch.save(checkpoint, self.filepath)
            if self.verbose:
                print(f"[BestModel] Saved to {self.filepath} (Score: {current_score:.4f})")
        return is_best

def load_checkpoint(filepath, model, device='cpu'):
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint.get('epoch', 0)
    return 0
