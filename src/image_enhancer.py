import cv2
import numpy as np

# =========================================================================
# CONFIGURATION PARAMETERS FOR ADAPTIVE IMAGE ENHANCEMENT
# =========================================================================
ENABLE_LOW_LIGHT_ENHANCEMENT = True
BRIGHTNESS_THRESHOLD = 85.0    # LAB L-channel mean intensity (0-255)
CONTRAST_THRESHOLD = 30.0      # Grayscale standard deviation intensity
GAMMA_VALUE = 1.3              # Power-law gamma expansion value
CLAHE_CLIP_LIMIT = 2.0         # CLAHE contrast limiting threshold
CLAHE_GRID_SIZE = (8, 8)       # Tile grid dimensions for local histogram equalization

# Pre-computed gamma lookup table
_INV_GAMMA = 1.0 / GAMMA_VALUE
_GAMMA_LUT = np.array([((i / 255.0) ** _INV_GAMMA) * 255 for i in range(256)]).astype("uint8")

def measure_frame_quality(frame_bgr):
    """
    Measures frame lighting quality:
    - mean_brightness: Mean pixel intensity in LAB L-channel (0-255)
    - contrast_std: Standard deviation of grayscale intensities
    - noise_blur_score: Variance of Laplacian
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return 0.0, 0.0, 0.0

    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    mean_brightness = float(np.mean(l_channel))

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    contrast_std = float(np.std(gray))
    noise_blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return mean_brightness, contrast_std, noise_blur_score

def adaptive_enhance_frame(frame_bgr):
    """
    Adaptively enhances frame only when low brightness or contrast is detected.
    Preserves hand edges and color accuracy without unnecessary overhead.
    
    Returns:
        enhanced_bgr (ndarray): Processed image or original pass-through
        was_enhanced (bool): True if enhancement was applied, False if passed through
    """
    if not ENABLE_LOW_LIGHT_ENHANCEMENT or frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr, False

    mean_brightness, contrast_std, _ = measure_frame_quality(frame_bgr)

    # If lighting is normal, pass through with zero computational overhead
    if mean_brightness >= BRIGHTNESS_THRESHOLD and contrast_std >= CONTRAST_THRESHOLD:
        return frame_bgr, False

    # 1. Fast Lookup Table Gamma Correction
    gamma_corr = cv2.LUT(frame_bgr, _GAMMA_LUT)

    # 2. CLAHE on LAB L-channel (preserves color fidelity & hand contours)
    lab = cv2.cvtColor(gamma_corr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
    l_enhanced = clahe.apply(l)

    lab_enhanced = cv2.merge((l_enhanced, a, b))
    enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return enhanced_bgr, True

if __name__ == '__main__':
    dummy = np.full((480, 640, 3), 40, dtype=np.uint8) # Dark dummy frame
    b, c, n = measure_frame_quality(dummy)
    print(f"Dark Frame Metrics -> Brightness: {b:.1f}, Contrast: {c:.1f}, Laplacian Var: {n:.1f}")
    
    enhanced, was_enhanced = adaptive_enhance_frame(dummy)
    b_enh, c_enh, _ = measure_frame_quality(enhanced)
    print(f"Enhanced Metrics   -> Brightness: {b_enh:.1f}, Contrast: {c_enh:.1f}, Was Enhanced: {was_enhanced}")
