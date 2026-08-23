import os
import sys
import numpy as np

def compute_angle(a, b, c):
    """
    Computes angle at vertex b given 3D points a, b, c (in radians).
    """
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0
    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.arccos(cosine))

def extract_hand_angles(hand_coords):
    """
    Extracts 5 finger joint flexion angles for a single hand (21x3 coords).
    Fingers: Thumb, Index, Middle, Ring, Pinky
    Landmark trios: (0,2,4), (0,5,8), (0,9,12), (0,13,16), (0,17,20)
    """
    hand = hand_coords.reshape(21, 3)
    if np.abs(hand).sum() < 1e-6:
        return np.zeros(5, dtype=np.float32)
        
    angles = []
    finger_trios = [
        (0, 2, 4),   # Thumb
        (0, 5, 8),   # Index
        (0, 9, 12),  # Middle
        (0, 13, 16), # Ring
        (0, 17, 20)  # Pinky
    ]
    for p1, p2, p3 in finger_trios:
        ang = compute_angle(hand[p1], hand[p2], hand[p3])
        angles.append(ang)
    return np.array(angles, dtype=np.float32)

def extract_hand_orientation(hand_coords):
    """
    Extracts palm normal vector (3D) and wrist-to-MCP9 direction vector (3D) for a hand.
    Returns 6D vector.
    """
    hand = hand_coords.reshape(21, 3)
    if np.abs(hand).sum() < 1e-6:
        return np.zeros(6, dtype=np.float32)
        
    v1 = hand[5] - hand[0]  # Wrist to Index MCP
    v2 = hand[17] - hand[0] # Wrist to Pinky MCP
    normal = np.cross(v1, v2)
    norm_len = np.linalg.norm(normal)
    normal = normal / norm_len if norm_len > 1e-6 else np.zeros(3, dtype=np.float32)
    
    dir_mcp = hand[9] - hand[0]
    dir_len = np.linalg.norm(dir_mcp)
    dir_mcp = dir_mcp / dir_len if dir_len > 1e-6 else np.zeros(3, dtype=np.float32)
    
    return np.concatenate([normal, dir_mcp])

def compute_enhanced_geometric_features(raw_126_seq):
    """
    Computes Enhanced Geometric Features for a (30, 126) landmark sequence.
    Output dim:
    - 126D position normalized
    - 126D velocity normalized
    - 4D inter-hand keypoint distances
    - 10D finger joint angles (5 left + 5 right)
    - 12D hand orientation vectors (6 left + 6 right)
    - 2D motion velocity magnitudes (left + right L2 velocity)
    Total Dim = 126 + 126 + 4 + 10 + 12 + 2 = 280D
    """
    T, D = raw_126_seq.shape
    
    # 1. Wrist-centered & palm scale normalization
    pos_norm = np.zeros_like(raw_126_seq)
    for t in range(T):
        frame = raw_126_seq[t]
        left = frame[:63].reshape(21, 3).copy()
        right = frame[63:].reshape(21, 3).copy()
        
        if np.abs(left).sum() > 1e-6:
            left -= left[0]
            dist = np.linalg.norm(left[9]) if np.linalg.norm(left[9]) > 1e-4 else np.max(np.linalg.norm(left, axis=1))
            if dist > 1e-4: left /= dist
            
        if np.abs(right).sum() > 1e-6:
            right -= right[0]
            dist = np.linalg.norm(right[9]) if np.linalg.norm(right[9]) > 1e-4 else np.max(np.linalg.norm(right, axis=1))
            if dist > 1e-4: right /= dist
            
        pos_norm[t] = np.concatenate([left.flatten(), right.flatten()])
        
    # 2. Causal First-Order Velocity
    vel_norm = np.zeros_like(pos_norm)
    for t in range(1, T):
        vel_norm[t] = pos_norm[t] - pos_norm[t-1]
        
    # 3. Geometric Extras per frame
    geo_extras = []
    for t in range(T):
        frame = raw_126_seq[t]
        left = frame[:63].reshape(21, 3)
        right = frame[63:].reshape(21, 3)
        
        left_present = (np.abs(left).sum() > 1e-6)
        right_present = (np.abs(right).sum() > 1e-6)
        
        # Inter-hand distances (4D)
        if left_present and right_present:
            dist_wrist = float(np.linalg.norm(left[0] - right[0]))
            dist_index = float(np.linalg.norm(left[8] - right[8]))
            dist_palm = float(np.linalg.norm(left[9] - right[9]))
            dist_thumb = float(np.linalg.norm(left[4] - right[4]))
        else:
            dist_wrist, dist_index, dist_palm, dist_thumb = 0.0, 0.0, 0.0, 0.0
            
        inter_hand_dist = np.array([dist_wrist, dist_index, dist_palm, dist_thumb], dtype=np.float32)
        
        # Finger angles (10D)
        left_angles = extract_hand_angles(left)
        right_angles = extract_hand_angles(right)
        finger_angles = np.concatenate([left_angles, right_angles])
        
        # Hand orientation (12D)
        left_orient = extract_hand_orientation(left)
        right_orient = extract_hand_orientation(right)
        orientations = np.concatenate([left_orient, right_orient])
        
        # Motion velocity magnitudes (2D)
        left_vel = vel_norm[t, :63]
        right_vel = vel_norm[t, 63:]
        motion_mag = np.array([np.linalg.norm(left_vel), np.linalg.norm(right_vel)], dtype=np.float32)
        
        extra_t = np.concatenate([inter_hand_dist, finger_angles, orientations, motion_mag])
        geo_extras.append(extra_t)
        
    geo_extras = np.array(geo_extras, dtype=np.float32) # (30, 28)
    
    # Concatenate 126D pos + 126D vel + 28D geometric = 280D
    feat_280 = np.concatenate([pos_norm, vel_norm, geo_extras], axis=-1)
    return feat_280

if __name__ == '__main__':
    dummy = np.random.randn(30, 126).astype(np.float32)
    out = compute_enhanced_geometric_features(dummy)
    print(f"Geometric Features Extracted Shape: {out.shape} (Expected 30, 280)")
