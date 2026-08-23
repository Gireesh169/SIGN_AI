import os
import sys
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BASE_DIR, DATA_DIR, MODELS_DIR, RANDOM_SEED, BATCH_SIZE, DEVICE
from hand_crop_extractor import HandCropCNNEncoder

WORD_FRAMES_DIR = os.path.join(BASE_DIR, 'ISL_CSLRT_Corpus', 'Frames_Word_Level')
SAVE_CNN_PATH = os.path.join(MODELS_DIR, 'word_finetuned_mobilenet.pth')
WORD_MAPPING_PATH = os.path.join(MODELS_DIR, 'word_label_mapping.json')

# Data Augmentation for Word-Level Training Split ONLY
train_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class WordSignDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

class WordCNNClassifier(nn.Module):
    def __init__(self, num_classes=114, embed_dim=64):
        super(WordCNNClassifier, self).__init__()
        self.encoder = HandCropCNNEncoder(embed_dim=embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (B, 3, 112, 112)
        # Create sequence shape (B, T=1, N=2, 3, 112, 112)
        single_crop = x.unsqueeze(1).unsqueeze(2) # (B, 1, 1, 3, 112, 112)
        crops = torch.cat([single_crop, single_crop], dim=2) # (B, 1, 2, 3, 112, 112)
        emb = self.encoder(crops).squeeze(1) # (B, 64)
        logits = self.fc(emb)
        return logits

def train_word_level_cnn():
    print("\n" + "="*70)
    print("=== STARTING WORD-LEVEL VISUAL ENCODER FINE-TUNING ===")
    print("="*70)

    if not os.path.exists(WORD_FRAMES_DIR):
        print(f"[Error] Word frames directory not found: {WORD_FRAMES_DIR}")
        return

    word_classes = sorted([d for d in os.listdir(WORD_FRAMES_DIR) if os.path.isdir(os.path.join(WORD_FRAMES_DIR, d)) and not d.startswith('.')])
    word_label_map = {cls_name: i for i, cls_name in enumerate(word_classes)}

    with open(WORD_MAPPING_PATH, 'w') as f:
        json.dump(word_label_map, f, indent=2)

    image_paths = []
    labels = []

    for cls_name in word_classes:
        cls_dir = os.path.join(WORD_FRAMES_DIR, cls_name)
        img_files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for p in img_files:
            image_paths.append(p)
            labels.append(word_label_map[cls_name])

    print(f"Found {len(image_paths)} word images across {len(word_classes)} word gesture classes.")

    # 80 / 20 Train / Val Split
    np.random.seed(RANDOM_SEED)
    indices = np.arange(len(image_paths))
    np.random.shuffle(indices)
    split_idx = int(0.8 * len(image_paths))

    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    train_paths = [image_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_paths = [image_paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    ds_train = WordSignDataset(train_paths, train_labels, transform=train_transform)
    ds_val = WordSignDataset(val_paths, val_labels, transform=val_transform)

    dl_train = DataLoader(ds_train, batch_size=32, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=32, shuffle=False)

    model = WordCNNClassifier(num_classes=len(word_classes), embed_dim=64).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, 9):
        model.train()
        train_loss, train_correct = 0.0, 0
        for imgs, lbls in dl_train:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == lbls).sum().item()
            train_loss += loss.item() * len(lbls)

        train_acc = train_correct / len(ds_train)

        # Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for imgs, lbls in dl_val:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                logits = model(imgs)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == lbls).sum().item()

        val_acc = val_correct / len(ds_val)
        print(f"Epoch {epoch}/8 -> Train Acc: {train_acc*100:.2f}%, Val Acc: {val_acc*100:.2f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.encoder.state_dict())

    if best_state:
        torch.save(best_state, SAVE_CNN_PATH)
        print(f"\n[Saved] Fine-tuned MobileNetV3 encoder saved to: {SAVE_CNN_PATH}")

if __name__ == '__main__':
    train_word_level_cnn()
