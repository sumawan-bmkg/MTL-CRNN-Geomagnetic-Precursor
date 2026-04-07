import torch
import torch.nn as nn
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
import torch.nn.functional as F

class ScalogramV2Model(nn.Module):
    """
    ScalogramV2 Hierarchical Multi-Task Learning (MTL) Architecture.
    
    This model combines a Convolutional Neural Network (EfficientNet-B1) for spatial 
    feature extraction from geomagnetic scalograms with a Bidirectional GRU (BiGRU) 
    for temporal reasoning. It utilizes three task-specific heads to perform 
    precursor detection, magnitude grading, and epicenter azimuth localization.
    
    Attributes:
        backbone: EfficientNet-B1 feature extractor.
        freq_pool: Adaptive pooling to focus on temporal evolution across frequencies.
        gru: Bidirectional GRU to capture precursor signal dynamics.
        fc_binary: Output head for binary classification (Normal vs Precursor).
        fc_magnitude: Output head for magnitude grading (3 categories).
        fc_azimuth: Sequential head for circular azimuth regression (sin/cos).
    """
    def __init__(self, num_classes=2, backbone='efficientnet_b1', pretrained=True):
        """
        Initializes the MTL-CRNN model.
        
        Args:
            num_classes (int): Number of classes for the primary detection task.
            backbone (str): Name of the CNN backbone architecture.
            pretrained (bool): Whether to use ImageNet pre-trained weights.
        """
        super(ScalogramV2Model, self).__init__()
        
        # Load backbone
        weights = EfficientNet_B1_Weights.DEFAULT if pretrained else None
        if backbone == 'efficientnet_b1':
            self.backbone = efficientnet_b1(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
            
        # Get in_features for translation to RNN
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.features = self.backbone.features
        
        # Frequency pooling (H dimension) to compress into temporal sequence (W dimension)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None)) 
        
        # Bidirectional GRU for sequential modeling
        self.hidden_size = 128
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(p=0.3)
        
        # --- Task-Specific Output Heads ---
        # 1. Detection Phase (Binary: Normal/Precursor)
        self.fc_binary = nn.Linear(self.hidden_size * 2, num_classes)
        
        # 2. Magnitude Grading (Categorical: 3 Levels)
        self.fc_magnitude = nn.Linear(self.hidden_size * 2, 3)
        
        # 3. Azimuth Localization (Circular Regression: [sin, cos])
        self.fc_azimuth = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 2)
        )
        
    def forward(self, x):
        """
        Forward pass for Multi-Task inference.
        
        Args:
            x (torch.Tensor): Input scalogram tensor of shape [B, 3, H, W].
            
        Returns:
            out_b (torch.Tensor): Logits for precursor detection [B, 2].
            out_m (torch.Tensor): Logits for magnitude grading [B, 3].
            out_a_sincos (torch.Tensor): L2-normalized [sin, cos] vector for azimuth [B, 2].
        """
        # Feature Extraction -> [B, 1280, H, W]
        feat_map = self.features(x)
        
        # Squeeze frequency axis -> [B, 1280, 1, W]
        temporal_feat = self.freq_pool(feat_map)
        
        # Reshape for GRU -> [B, W, 1280]
        temporal_feat = temporal_feat.squeeze(2).transpose(1, 2)
        
        # Temporal Modeling
        gru_out, hidden = self.gru(temporal_feat) 
        
        # Hidden State Fusion (Forward + Backward)
        final_state = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1) 
        final_state = self.dropout(final_state)
        
        # Parallel Multi-Task Heads
        out_b = self.fc_binary(final_state)
        out_m = self.fc_magnitude(final_state)
        out_a_raw = self.fc_azimuth(final_state)
        
        # Vector Normalization for Von Mises Circularity
        out_a_sincos = F.normalize(out_a_raw, p=2, dim=1)
        
        return out_b, out_m, out_a_sincos
