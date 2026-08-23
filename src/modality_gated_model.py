import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttentionPooling(nn.Module):
    def __init__(self, feature_dim):
        super(TemporalAttentionPooling, self).__init__()
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
        return context

class HandMotionBranch(nn.Module):
    """
    Branch 1: Hand Motion (252D -> 64D Embedding)
    """
    def __init__(self, input_dim=252, hidden_dim=32, embed_dim=64, dropout=0.3):
        super(HandMotionBranch, self).__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.attn = TemporalAttentionPooling(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (B, 30, 252)
        h = self.norm(self.proj(x))
        out, _ = self.bilstm(h)
        emb = self.attn(out) # (B, 64)
        return self.dropout(emb)

class FaceBranch(nn.Module):
    """
    Branch 2: Facial Features (16D -> 64D Embedding)
    """
    def __init__(self, input_dim=16, hidden_dim=16, embed_dim=64, dropout=0.3):
        super(FaceBranch, self).__init__()
        self.proj = nn.Linear(input_dim, 32)
        self.norm = nn.LayerNorm(32)
        self.bigru = nn.GRU(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.out_proj = nn.Linear(hidden_dim * 2, embed_dim)
        self.attn = TemporalAttentionPooling(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (B, 30, 16)
        f = self.norm(self.proj(x))
        out, _ = self.bigru(f)
        out = self.out_proj(out)
        emb = self.attn(out) # (B, 64)
        return self.dropout(emb)

class PoseBranch(nn.Module):
    """
    Branch 3: Upper-Body Pose Features (24D -> 64D Embedding)
    """
    def __init__(self, input_dim=24, hidden_dim=16, embed_dim=64, dropout=0.3):
        super(PoseBranch, self).__init__()
        self.proj = nn.Linear(input_dim, 32)
        self.norm = nn.LayerNorm(32)
        self.bigru = nn.GRU(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.out_proj = nn.Linear(hidden_dim * 2, embed_dim)
        self.attn = TemporalAttentionPooling(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (B, 30, 24)
        p = self.norm(self.proj(x))
        out, _ = self.bigru(p)
        out = self.out_proj(out)
        emb = self.attn(out) # (B, 64)
        return self.dropout(emb)

class ModalityGatedFusionModel(nn.Module):
    """
    3-Branch Modality Gated Multimodal Fusion Classifier.
    """
    def __init__(
        self,
        hand_dim=252,
        face_dim=16,
        pose_dim=24,
        embed_dim=64,
        num_classes=97,
        use_face=True,
        use_pose=True,
        dropout=0.3
    ):
        super(ModalityGatedFusionModel, self).__init__()
        self.use_face = use_face
        self.use_pose = use_pose
        
        self.hand_branch = HandMotionBranch(input_dim=hand_dim, embed_dim=embed_dim, dropout=dropout)
        
        if use_face:
            self.face_branch = FaceBranch(input_dim=face_dim, embed_dim=embed_dim, dropout=dropout)
        if use_pose:
            self.pose_branch = PoseBranch(input_dim=pose_dim, embed_dim=embed_dim, dropout=dropout)
            
        num_modalities = 1 + (1 if use_face else 0) + (1 if use_pose else 0)
        
        # Learnable Modality Gating Module
        self.gate_layer = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, 32),
            nn.Tanh(),
            nn.Linear(32, num_modalities)
        )
        
        # Final Classification Head
        self.fc1 = nn.Linear(embed_dim, 64)
        self.norm = nn.LayerNorm(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x_hand, x_face=None, x_pose=None):
        emb_h = self.hand_branch(x_hand) # (B, 64)
        embs = [emb_h]
        
        if self.use_face and x_face is not None:
            emb_f = self.face_branch(x_face)
            embs.append(emb_f)
        if self.use_pose and x_pose is not None:
            emb_p = self.pose_branch(x_pose)
            embs.append(emb_p)
            
        if len(embs) == 1:
            fused_emb = emb_h
        else:
            concat_embs = torch.cat(embs, dim=1) # (B, 64 * K)
            gate_logits = self.gate_layer(concat_embs) # (B, K)
            weights = torch.softmax(gate_logits, dim=1) # (B, K) normalized
            
            fused_emb = torch.zeros_like(emb_h)
            for i, emb in enumerate(embs):
                fused_emb += weights[:, i:i+1] * emb
                
        out = self.fc1(fused_emb)
        out = self.norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)
        return logits

if __name__ == '__main__':
    model = ModalityGatedFusionModel(use_face=True, use_pose=True)
    dh = torch.randn(2, 30, 252)
    df = torch.randn(2, 30, 16)
    dp = torch.randn(2, 30, 24)
    out = model(dh, df, dp)
    print(f"Modality Gated Fusion Logits Shape: {out.shape} (Expected: (2, 97))")
