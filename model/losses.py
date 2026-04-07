import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss to mitigate class imbalance in Precursor Detection.
    """
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()

class VonMisesLoss(nn.Module):
    """
    Von Mises Negative Log-Likelihood for Circular Azimuth Regression.
    """
    def __init__(self, kappa=5.0):
        super(VonMisesLoss, self).__init__()
        self.kappa = kappa

    def forward(self, pred_sincos, target_rad):
        # target_rad expected in radians
        target_sin = torch.sin(target_rad)
        target_cos = torch.cos(target_rad)
        
        # Cosine similarity between prediction and target vectors
        cos_diff = pred_sincos[:, 0] * target_sin + pred_sincos[:, 1] * target_cos
        loss = self.kappa - (self.kappa * cos_diff)
        return loss.mean()

class GeomagneticPenaltyLoss(nn.Module):
    """
    Specialized Loss for weighting Normal conditions during Solar Storms.
    """
    def __init__(self, base_loss_fn, solar_penalty=5.0):
        super(GeomagneticPenaltyLoss, self).__init__()
        self.base_loss_fn = base_loss_fn
        self.solar_penalty = solar_penalty

    def forward(self, pred, target, is_solar_storm):
        loss_unweighted = self.base_loss_fn(pred, target)
        
        # Apply penalty for False Positives during Solar Storms
        penalty_mask = torch.ones_like(loss_unweighted)
        penalty_mask[(target == 0) & (is_solar_storm == 1)] = self.solar_penalty
        
        return (loss_unweighted * penalty_mask).mean()
