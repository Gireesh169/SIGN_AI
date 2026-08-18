import os
import torch
import numpy as np

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR = os.path.join(BASE_DIR, 'ISL_CSLRT_Corpus')
VIDEOS_DIR = os.path.join(CORPUS_DIR, 'Videos_Sentence_Level')
FRAMES_DIR = os.path.join(CORPUS_DIR, 'Frames_Sentence_Level')

DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Data files
PROGRESS_FILE = os.path.join(DATA_DIR, 'extraction_progress.json')
PARTIAL_X_FILE = os.path.join(DATA_DIR, 'x_features_partial.npy')
PARTIAL_Y_FILE = os.path.join(DATA_DIR, 'y_labels_partial.npy')

X_FEATURES_FILE = os.path.join(DATA_DIR, 'x_features.npy')
Y_LABELS_FILE = os.path.join(DATA_DIR, 'y_labels.npy')
LABEL_MAPPING_FILE = os.path.join(DATA_DIR, 'sentence_labels.json')
FAILED_VIDEOS_FILE = os.path.join(DATA_DIR, 'failed_videos.csv')

# Model output files
MODEL_PATH = os.path.join(MODELS_DIR, 'bilstm_gru_model.pth')
BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.pth')
EVAL_METRICS_PATH = os.path.join(MODELS_DIR, 'eval_metrics.json')
CONFUSION_MATRIX_PATH = os.path.join(MODELS_DIR, 'confusion_matrix.png')
TRAINING_HISTORY_PATH = os.path.join(MODELS_DIR, 'training_history.png')

# Model Parameters
SEQUENCE_LENGTH = 30
LANDMARKS_PER_HAND = 21
NUM_HANDS = 2
COORDS_PER_LANDMARK = 3
FEATURE_DIM = NUM_HANDS * LANDMARKS_PER_HAND * COORDS_PER_LANDMARK  # 126

# Training Hyperparameters
RANDOM_SEED = 42
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 32
NUM_BILSTM_LAYERS = 1
NUM_GRU_LAYERS = 1
DROPOUT = 0.30

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def normalize_sequence_features(seq_arr):
    """
    Relative wrist landmark centering AND palm scale normalization for a (T, 126) sequence or (N, T, 126) tensor.
    Centers Left hand landmarks relative to Left Wrist and scales by palm distance.
    Centers Right hand landmarks relative to Right Wrist and scales by palm distance.
    """
    is_3d = (seq_arr.ndim == 3)
    X_in = seq_arr if is_3d else seq_arr[np.newaxis, ...]
    N, T, D = X_in.shape
    X_norm = X_in.copy()
    
    for i in range(N):
        for t in range(T):
            frame = X_norm[i, t]
            left = frame[:63].reshape(21, 3)
            right = frame[63:].reshape(21, 3)
            
            if np.abs(left).sum() > 1e-6:
                # Wrist centering
                left = left - left[0]
                # Scale normalization (wrist to middle MCP landmark 9 distance, or max distance)
                dist = np.linalg.norm(left[9]) if np.linalg.norm(left[9]) > 1e-4 else np.max(np.linalg.norm(left, axis=1))
                if dist > 1e-4:
                    left = left / dist
                    
            if np.abs(right).sum() > 1e-6:
                # Wrist centering
                right = right - right[0]
                # Scale normalization (wrist to middle MCP landmark 9 distance, or max distance)
                dist = np.linalg.norm(right[9]) if np.linalg.norm(right[9]) > 1e-4 else np.max(np.linalg.norm(right, axis=1))
                if dist > 1e-4:
                    right = right / dist
                
            X_norm[i, t] = np.concatenate([left.flatten(), right.flatten()])
            
    return X_norm if is_3d else X_norm[0]

