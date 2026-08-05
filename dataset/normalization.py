import numpy as np


def get_data_stats(data: np.ndarray) -> dict[str, np.ndarray]:
    """Computes the min and max for each feature dimension across the entire dataset."""
    return {
        "min": np.min(data, axis=0),
        "max": np.max(data, axis=0)
    }
    

def normalize_data(data: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    """Normalizes data to the [-1, 1] range using the computed statistics."""
    d_min = stats["min"]
    d_max = stats["max"]
    
    denominator = d_max - d_min
    denominator = np.where(denominator < 1e-8, 1e-8, denominator)
    
    ndata = 2.0 * (data - d_min) / denominator - 1.0
    return ndata


def unnormalize_data(ndata: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    """Reverses the [-1, 1] normalization back to the original environment scale."""
    d_min = stats["min"]
    d_max = stats["max"]
    
    denominator = d_max - d_min
    
    undata = (denominator * (ndata + 1.0) / 2) + d_min
    return undata