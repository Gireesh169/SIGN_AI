import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BASE_DIR, CORPUS_DIR, MODELS_DIR, DEVICE
from hand_crop_extractor import HandCropCNNEncoder

WORD_FRAMES_DIR = os.path.join(CORPUS_DIR, 'Frames_Word_Level')
WORD_MODEL_PATH = os.path.join(MODELS_DIR, 'word_pretrained_cnn.pth')

class WordLevelHandDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        
        if not os.path.exists(root_dir):
            print(f"[Word Pretrain] Directory {root_dir} not found. Using dummy dataset for verification.")
            return
            
        classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        
        for cls_name in classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for root, _, files in os.walk(cls_dir):
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(root, f))
                        self.labels.append(self.class_to_idx[cls_name])
                        
        print(f"[Word Pretrain Dataset] Found {len(self.image_paths)} word-level hand images across {len(classes)} word classes.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((112, 112, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (112, 112))
            
        img_t = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        if self.transform:
            img_t = self.transform(img_t)
            
        return img_t, self.labels[idx]

def pretrain_word_level_cnn(epochs=5, batch_size=32):
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = WordLevelHandDataset(WORD_FRAMES_DIR, transform=transform)
    
    if len(dataset) == 0:
        print("[Word Pretrain] No word-level images available to pretrain. Saving default CNN weights.")
        encoder = HandCropCNNEncoder(embed_dim=64)
        torch.save(encoder.state_dict(), WORD_MODEL_PATH)
        return WORD_MODEL_PATH
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    num_classes = len(dataset.class_to_idx)
    
    encoder = HandCropCNNEncoder(embed_dim=64).to(DEVICE)
    classifier_head = nn.Linear(64, num_classes).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(list(encoder.parameters()) + list(classifier_head.parameters()), lr=1e-3, weight_decay=1e-4)
    
    print(f"\n--- Pretraining Hand Crop CNN Encoder on {num_classes} Word Classes ({epochs} Epochs) ---")
    encoder.train()
    classifier_head.train()
    
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Format to (B, 1, 2, 3, 112, 112) dummy sequence format for HandCropCNNEncoder
            # Duplicate image for Left and Right hand inputs
            B = imgs.shape[0]
            dummy_seq = imgs.unsqueeze(1).unsqueeze(2).repeat(1, 1, 2, 1, 1, 1) # (B, 1, 2, 3, 112, 112)
            
            optimizer.zero_grad()
            embed = encoder(dummy_seq).squeeze(1) # (B, 64)
            logits = classifier_head(embed) # (B, num_classes)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * B
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += B
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total * 100.0
        print(f"Epoch [{epoch}/{epochs}] Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
        
    torch.save(encoder.state_dict(), WORD_MODEL_PATH)
    print(f"[Word Pretrain] Pretrained weights saved to: {WORD_MODEL_PATH}\n")
    return WORD_MODEL_PATH

if __name__ == '__main__':
    pretrain_word_level_cnn(epochs=3)
