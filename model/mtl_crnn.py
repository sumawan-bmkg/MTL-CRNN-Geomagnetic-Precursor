import torch
import torch.nn as nn
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
import torch.nn.functional as F

class ScalogramV2Model(nn.Module):
    """
    MTL-CRNN Architecture: EfficientNet-B1 + BiGRU
    Optimized for Geomagnetic Spatio-Temporal Tensors.
    """
    def __init__(self, num_classes=2, backbone='efficientnet_b1', pretrained=True):
        super(ScalogramV2Model, self).__init__()
        
        # Load backbone
        weights = EfficientNet_B1_Weights.DEFAULT if pretrained else None
        if backbone == 'efficientnet_b1':
            self.backbone = efficientnet_b1(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
            
        # Get in_features for the BiGRU
        in_features = self.backbone.classifier[1].in_features
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        self.features = self.backbone.features
        
        # Pool across frequency (H) to isolate temporal evolution (W)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None)) 
        
        # Bidirectional GRU
        self.hidden_size = 128
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(p=0.3)
        
        # --- Task-Specific Heads ---
        # 1. Detection Phase (Binary: Normal/Precursor)
        self.fc_binary = nn.Linear(self.hidden_size * 2, num_classes)
        
        # 2. Magnitude Grading (Classification: 3 Levels)
        self.fc_magnitude = nn.Linear(self.hidden_size * 2, 3)
        
        # 3. Azimuth Localization (Von Mises sincos Regression)
        self.fc_azimuth = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 2)
        )
        
    def forward(self, x):
        # Extract visual feature map -> [B, 1280, H, W]
        feat_map = self.features(x)
        
        # Frequency pooling to emphasize temporal patterns -> [B, 1280, 1, W]
        temporal_feat = self.freq_pool(feat_map)
        
        # Prep for Sequential BiGRU -> [B, W, 1280]
        temporal_feat = temporal_feat.squeeze(2).transpose(1, 2)
        
        # Temporal reasoning
        gru_out, hidden = self.gru(temporal_feat) 
        
        # Concatenate forward and backward final hidden states
        final_state = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1) 
        final_state = self.dropout(final_state)
        
        # Multi-Task Outputs
        out_b = self.fc_binary(final_state)
        out_m = self.fc_magnitude(final_state)
        out_a_raw = self.fc_azimuth(final_state)
        
        # Force L2 Unit Vector for Von Mises Azimuth
        out_a_sincos = F.normalize(out_a_raw, p=2, dim=1)
        
        return out_b, out_m, out_a_sincos
