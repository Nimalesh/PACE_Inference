import numpy as np
import cv2

def keep_largest_cc(mask_np: np.ndarray) -> np.ndarray:
    if mask_np.max() == 0:
        return mask_np
    
    mask_bin = (mask_np > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    
    if num_labels <= 1:
        return mask_np
        
    largest_component_index = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    
    clean_mask = np.zeros_like(mask_np)
    clean_mask[labels == largest_component_index] = 255
    
    return clean_mask