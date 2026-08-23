import torch
import torch.nn as nn
import torch.nn.functional as F
from hand_crop_extractor import HandCropCNNEncoder

class TemporalAttention(nn.Module):
    def __init__(self, feature_dim):
        super(TemporalAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(feature_dim, max(16, feature_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(16, feature_dim // 2), 1)
        )
        
    def forward(self, x):
        # x: (batch_size, seq_len, feature_dim)
        scores = self.attn(x)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(x * weights, dim=1)
        return context, weights

class MotionBranch(nn.Module):
    """
    Branch 1: Landmark Position & Velocity (252D/280D) Temporal Feature Extractor.
    """
    def __init__(self, input_dim=252, hidden_dim=32, dropout=0.30):
        super(MotionBranch, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1
        )
        self.conv_bn = nn.BatchNorm1d(hidden_dim)
        self.conv_act = nn.SiLU()
        self.conv_drop = nn.Dropout(0.20)
        
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim * 2)
        
        self.bigru = nn.GRU(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim * 2)
        
        self.attention = TemporalAttention(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch_size, 30, input_dim)
        x_conv = x.transpose(1, 2)
        x_conv = self.conv_drop(self.conv_act(self.conv_bn(self.conv1d(x_conv))))
        x_conv = x_conv.transpose(1, 2)
        
        lstm_out, _ = self.bilstm(x_conv)
        lstm_out = self.norm1(lstm_out)
        
        gru_out, _ = self.bigru(lstm_out)
        gru_out = self.norm2(gru_out)
        
        attn_out, _ = self.attention(gru_out) # (batch_size, hidden_dim*2 = 64)
        mean_pool = gru_out.mean(dim=1)       # (batch_size, hidden_dim*2 = 64)
        
        motion_feat = torch.cat([attn_out, mean_pool], dim=1) # (batch_size, 128)
        return self.dropout(motion_feat)

class VisualBranch(nn.Module):
    """
    Branch 2: Hand-Crop RGB Visual Feature Extractor (MobileNetV3 + Temporal BiLSTM/BiGRU).
    """
    def __init__(self, embed_dim=64, hidden_dim=32, dropout=0.30):
        super(VisualBranch, self).__init__()
        self.cnn_encoder = HandCropCNNEncoder(embed_dim=embed_dim, dropout=dropout)
        
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim * 2)
        
        self.bigru = nn.GRU(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim * 2)
        
        self.attention = TemporalAttention(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x_crops):
        # x_crops: (batch_size, 30, 2, 3, 112, 112)
        embed_seq = self.cnn_encoder(x_crops) # (batch_size, 30, 64)
        
        lstm_out, _ = self.bilstm(embed_seq)
        lstm_out = self.norm1(lstm_out)
        
        gru_out, _ = self.bigru(lstm_out)
        gru_out = self.norm2(gru_out)
        
        attn_out, _ = self.attention(gru_out)
        mean_pool = gru_out.mean(dim=1)
        
        visual_feat = torch.cat([attn_out, mean_pool], dim=1) # (batch_size, 128)
        return self.dropout(visual_feat)

class GatedMultimodalModel(nn.Module):
    """
    Two-Branch Gated Multimodal Architecture (Motion + Hand-Crop Visual Branch).
    """
    def __init__(self, motion_dim=252, visual_embed_dim=64, hidden_dim=32, num_classes=97, dropout=0.30):
        super(GatedMultimodalModel, self).__init__()
        self.motion_branch = MotionBranch(input_dim=motion_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.visual_branch = VisualBranch(embed_dim=visual_embed_dim, hidden_dim=hidden_dim, dropout=dropout)
        
        feature_dim = hidden_dim * 4 # 128
        
        # Learnable Gating Module
        self.gate_fc = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.Sigmoid()
        )
        
        # Classification Head
        self.fc1 = nn.Linear(feature_dim, 64)
        self.norm = nn.LayerNorm(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, motion_input, visual_input=None):
        """
        motion_input: (batch_size, 30, motion_dim)
        visual_input: (batch_size, 30, 2, 3, 112, 112) or None
        """
        f_motion = self.motion_branch(motion_input) # (batch_size, 128)
        
        if visual_input is not None:
            f_visual = self.visual_branch(visual_input) # (batch_size, 128)
            
            # Gated Fusion
            concat_feat = torch.cat([f_motion, f_visual], dim=1) # (batch_size, 256)
            alpha = self.gate_fc(concat_feat) # (batch_size, 128) in [0, 1]
            
            f_fused = alpha * f_motion + (1.0 - alpha) * f_visual # (batch_size, 128)
        else:
            f_fused = f_motion
            
        out = self.fc1(f_fused)
        out = self.norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)
        
        return logits

if __name__ == '__main__':
    model = GatedMultimodalModel(motion_dim=252, num_classes=97)
    dummy_motion = torch.randn(2, 30, 252)
    dummy_visual = torch.randn(2, 30, 2, 3, 112, 112)
    
    out_motion_only = model(dummy_motion)
    out_multimodal = model(dummy_motion, dummy_visual)
    
    print(f"Motion-only Logits Shape: {out_motion_only.shape} (Expected: (2, 97))")
    print(f"Multimodal Logits Shape : {out_multimodal.shape} (Expected: (2, 97))")
