import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    """
    Softmax Temporal Attention layer to dynamically weight key sign execution time steps.
    """
    def __init__(self, feature_dim):
        super(TemporalAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(feature_dim, max(16, feature_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(16, feature_dim // 2), 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, feature_dim)
        scores = self.attn(x)  # (batch_size, seq_len, 1)
        weights = torch.softmax(scores, dim=1)  # (batch_size, seq_len, 1)
        context = torch.sum(x * weights, dim=1)  # (batch_size, feature_dim)
        return context, weights

class BiLSTM_GRU_Model(nn.Module):
    def __init__(self, input_dim=252, hidden_dim=32, num_classes=101, num_bilstm_layers=1, num_gru_layers=1, dropout=0.30):
        super(BiLSTM_GRU_Model, self).__init__()
        
        # 1D Temporal Convolutional Feature Extractor
        # If input_dim is 252 (252D position+velocity), in_channels = input_dim (252) or input_dim*2 (504 for trained weights)
        combined_dim = input_dim * 2 if input_dim == 252 else input_dim * 2
        
        self.conv1d = nn.Conv1d(
            in_channels=combined_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1
        )
        self.conv_bn = nn.BatchNorm1d(hidden_dim)
        self.conv_act = nn.SiLU()
        self.conv_drop = nn.Dropout(0.2)

        # Bidirectional LSTM Layer
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_bilstm_layers,
            batch_first=True,
            bidirectional=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim * 2)
        
        # Bidirectional GRU Layer
        self.bigru = nn.GRU(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim * 2)
        
        # Temporal Attention Layer
        self.attention = TemporalAttention(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        
        # Compact Classification Head (Attention Context + Global Mean Pooling)
        self.fc1 = nn.Linear(hidden_dim * 4, 64)
        self.norm3 = nn.LayerNorm(64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len=30, input_dim=126 or 252)
        batch_size, seq_len, in_dim = x.shape
        
        if in_dim == self.conv1d.in_channels:
            # Features already contain position + velocity (e.g. 252D)
            feat = x
        elif in_dim * 2 == self.conv1d.in_channels:
            # 126D position features: calculate velocity delta_x(t) = x(t) - x(t-1)
            vel = torch.cat([torch.zeros_like(x[:, :1, :]), x[:, 1:, :] - x[:, :-1, :]], dim=1)
            feat = torch.cat([x, vel], dim=2)
        else:
            # Default velocity computation fallback
            vel = torch.cat([torch.zeros_like(x[:, :1, :]), x[:, 1:, :] - x[:, :-1, :]], dim=1)
            feat = torch.cat([x, vel], dim=2) if (in_dim * 2) == self.conv1d.in_channels else x

        # 1D Temporal Convolution
        feat_conv = feat.transpose(1, 2) # (batch_size, channels, 30)
        feat_conv = self.conv_drop(self.conv_act(self.conv_bn(self.conv1d(feat_conv))))
        feat_conv = feat_conv.transpose(1, 2) # (batch_size, 30, hidden_dim)
        
        # BiLSTM & BiGRU
        lstm_out, _ = self.bilstm(feat_conv) # (batch_size, 30, hidden_dim*2)
        lstm_out = self.norm1(lstm_out)
        
        gru_out, _ = self.bigru(lstm_out) # (batch_size, 30, hidden_dim*2)
        gru_out = self.norm2(gru_out)
        
        # Attention Pooling & Mean Pooling
        attn_out, _ = self.attention(gru_out) # (batch_size, hidden_dim*2)
        mean_pool = gru_out.mean(dim=1) # (batch_size, hidden_dim*2)
        
        combined = torch.cat([attn_out, mean_pool], dim=1) # (batch_size, hidden_dim*4)
        
        out = self.dropout(combined)
        out = self.fc1(out)
        out = self.norm3(out)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)
        
        return logits
