import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalSignModel(nn.Module):
    """
    Multimodal ISL Sentence Recognition Architecture fusing:
    1. RGB Visual Sequence Embeddings (MobileNetV3 feature vectors: 30 x 576)
    2. MediaPipe 3D Landmark Sequences (30 x 126)
    """
    def __init__(
        self,
        visual_dim=576,
        landmark_dim=126,
        landmark_embed_dim=128,
        fusion_dim=256,
        hidden_dim=32,
        num_classes=97,
        dropout=0.30
    ):
        super(MultimodalSignModel, self).__init__()

        # Landmark Encoder Branch
        self.landmark_norm = nn.LayerNorm(landmark_dim)
        self.landmark_proj = nn.Sequential(
            nn.Linear(landmark_dim, landmark_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Visual + Landmark Timestep Fusion
        total_input_dim = visual_dim + landmark_embed_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(total_input_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Temporal Sequence Layers (BiLSTM + GRU)
        self.bilstm = nn.LSTM(
            input_size=fusion_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.gru = nn.GRU(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Softmax Temporal Attention Mechanism
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # Final Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes)
        )

    def forward(self, x_rgb, x_landmarks):
        """
        x_rgb: (batch_size, 30, 576)
        x_landmarks: (batch_size, 30, 126)
        """
        # Landmark Encoding
        lm_norm = self.landmark_norm(x_landmarks)
        lm_emb = self.landmark_proj(lm_norm) # (batch_size, 30, 128)

        # Timestep Multimodal Feature Concatenation
        multimodal_cat = torch.cat([x_rgb, lm_emb], dim=-1) # (batch_size, 30, 704)
        fused_features = self.fusion_proj(multimodal_cat) # (batch_size, 30, 256)

        # Temporal Sequence Processing
        lstm_out, _ = self.bilstm(fused_features) # (batch_size, 30, 64)
        gru_out, _ = self.gru(lstm_out) # (batch_size, 30, 64)

        # Softmax Temporal Attention Weight Pooling
        attn_weights = self.attention_net(gru_out) # (batch_size, 30, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        context_vector = torch.sum(attn_weights * gru_out, dim=1) # (batch_size, 64)

        # Classification Logits
        logits = self.classifier(context_vector) # (batch_size, num_classes)
        return logits, attn_weights
