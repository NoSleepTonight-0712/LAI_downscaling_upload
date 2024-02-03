import numpy as np
import torch

GCM_SHAPE = (60, 90)
TARGET_SHAPE = tuple(np.array(GCM_SHAPE) * 16)
TARGET_SHAPE_CV = TARGET_SHAPE[::-1]

OLD_GCM_SHAPE = (60, 91)
OLD_TARGET_SHAPE = tuple(np.array(OLD_GCM_SHAPE) * 16)
OLD_TARGET_SHAPE_CV = OLD_TARGET_SHAPE[::-1]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class CalibrationType:
    Quantile = 1
    Replacement = 2
    Hybrid_Quantile_Replacement = 3
    Hybrid_Replacement_Quantile = 4
    Latest = 5
