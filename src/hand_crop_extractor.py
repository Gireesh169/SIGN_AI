import os
import sys
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

class HandCropCNNEncoder(nn.Module):
    """
    Lightweight MobileNetV3-Small Hand Crop Feature Extractor.
    Extracts 64D compact visual embeddings per frame from Left & Right hand RGB crops (112x112).
    """
    def __init__(self, embed_dim=64, dropout=0.3):
        super(HandCropCNNEncoder, self).__init__()
        # Pretrained MobileNetV3-Small backbone
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Backbone output feature dimension for MobileNetV3-Small is 576
        in_dim = 576 * 2  # Left + Right hand crops
        
        self.projection = nn.Sequential(
            nn.Linear(in_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Input shape: (batch_size, seq_len=30, num_hands=2, C=3, H=112, W=112)
        Output shape: (batch_size, seq_len=30, embed_dim=64)
        """
        B, T, N, C, H, W = x.shape
        # Flatten batch, sequence, and hands dimensions for CNN extraction
        x_flat = x.view(B * T * N, C, H, W)
        
        feat_flat = self.backbone(x_flat) # (B*T*N, 576, h, w)
        feat_flat = self.avgpool(feat_flat).view(B * T * N, 576) # (B*T*N, 576)
        
        # Reshape back to (B, T, 2 * 576)
        feat_seq = feat_flat.view(B, T, N * 576)
        
        # Compact projection to 64D
        embed_seq = self.projection(feat_seq) # (B, T, 64)
        return embed_seq

def crop_hand_region(frame_rgb, hand_coords_63, crop_size=112, margin=0.20):
    """
    Crops hand region from RGB frame using 21 3D landmarks (63 coords).
    Applies 20% margin padding, resizes to (112, 112), and normalizes.
    If hand is missing (coords sum ~ 0), returns black/zero image.
    """
    h_img, w_img, _ = frame_rgb.shape
    coords = hand_coords_63.reshape(21, 3)
    
    if np.abs(coords).sum() < 1e-6:
        return np.zeros((crop_size, crop_size, 3), dtype=np.float32)
        
    # Extract x and y pixel coordinates
    xs = coords[:, 0] * w_img
    ys = coords[:, 1] * h_img
    
    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)
    
    w_box = max(x_max - x_min, 10)
    h_box = max(y_max - y_min, 10)
    
    # Expand by 20% margin
    x_min = max(0, int(x_min - w_box * margin))
    x_max = min(w_img, int(x_max + w_box * margin))
    y_min = max(0, int(y_min - h_box * margin))
    y_max = min(h_img, int(y_max + h_box * margin))
    
    crop = frame_rgb[y_min:y_max, x_min:x_max]
    if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
        return np.zeros((crop_size, crop_size, 3), dtype=np.float32)
        
    crop_resized = cv2.resize(crop, (crop_size, crop_size))
    crop_norm = crop_resized.astype(np.float32) / 255.0
    return crop_norm

def extract_hand_crops_sequence(frames_rgb, raw_126_seq, crop_size=112):
    """
    Extracts Left & Right hand RGB crops for a sequence of 30 frames.
    Output shape: (30, 2, 3, 112, 112) PyTorch FloatTensor format.
    """
    seq_len = len(frames_rgb)
    crops_seq = []
    
    # Normalization transform
    norm_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    for t in range(seq_len):
        frame = frames_rgb[t]
        raw_126 = raw_126_seq[t]
        
        left_raw = raw_126[:63]
        right_raw = raw_126[63:]
        
        left_crop = crop_hand_region(frame, left_raw, crop_size=crop_size)
        right_crop = crop_hand_region(frame, right_raw, crop_size=crop_size)
        
        # Convert HWC to CHW PyTorch tensor
        left_t = torch.tensor(left_crop.transpose(2, 0, 1), dtype=torch.float32)
        right_t = torch.tensor(right_crop.transpose(2, 0, 1), dtype=torch.float32)
        
        left_t = norm_transform(left_t)
        right_t = norm_transform(right_t)
        
        crops_seq.append(torch.stack([left_t, right_t])) # (2, 3, 112, 112)
        
    return torch.stack(crops_seq) # (30, 2, 3, 112, 112)

if __name__ == '__main__':
    encoder = HandCropCNNEncoder(embed_dim=64)
    dummy_input = torch.randn(2, 30, 2, 3, 112, 112)
    with torch.no_grad():
        out = encoder(dummy_input)
    print(f"Hand Crop CNN Encoder Output Shape: {out.shape} (Expected: (2, 30, 64))")
