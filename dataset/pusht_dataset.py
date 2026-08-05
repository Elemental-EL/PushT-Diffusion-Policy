import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from dataset.normalization import get_data_stats, normalize_data


class PushTStateDataset(Dataset):
    """
    Dataset for the low-dimensional PushT environment, implementing action chunking for Diffusion Policy architectures.

    Loads the Zarr dataset into RAM, applies global Min-Max normalization, and pads episode boundaries to allow O(1) sliding window extraction during the training loop.

    Args:
        dataset_path (str): Filepath to the dataset directory.
        pred_horizon (int): Number of future action steps to predict (T_p).
        obs_horizon (int): Number of past observation steps to condition on (T_o).
        action_horizon (int): Number of actions executed per inference step (T_a).

    Attributes:
        stats (dict): Global min/max statistics used for un-normalizing outputs.
        episodes (list): Normalized and padded trajectory dictionaries.
        indices (list): Global mapping from DataLoader index to (episode_id, local_time).
    """
    def __init__(self, dataset_path: str, pred_horizon: int, obs_horizon: int, action_horizon: int):
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        
        dataset_root = zarr.open(dataset_path, 'r')
        state_data = dataset_root['data']['state'][:]
        action_data = dataset_root['data']['action'][:]
        episode_ends = dataset_root['meta']['episode_ends'][:]
        
        self.stats = {
            "state": get_data_stats(state_data),
            "action": get_data_stats(action_data)
        }
        
        norm_state = normalize_data(state_data, self.stats["state"])
        norm_action = normalize_data(action_data, self.stats["action"])
        
        self.episodes = []
        self.indices = []
        
        start_idx = 0
        for i, end_idx in enumerate(episode_ends):
            ep_state = norm_state[start_idx:end_idx]
            ep_action = norm_action[start_idx:end_idx]
            ep_length = len(ep_state)
            
            pad_state_start = np.repeat(ep_state[0:1], obs_horizon - 1, axis=0)
            padded_state = np.vstack([pad_state_start, ep_state])
            
            pad_action_end = np.repeat(ep_action[-1:], pred_horizon - 1, axis=0)
            padded_action = np.vstack([ep_action, pad_action_end])
            
            self.episodes.append({
                'state': padded_state,
                'action': padded_action
            })
            
            for t in range(ep_length):
                self.indices.append((i, t))
                
            start_idx = end_idx

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        """Fetches a single sliding window of observations and future actions."""
        ep_idx, t = self.indices[idx]
        ep = self.episodes[ep_idx]
        
        state_seq = ep['state'][t : t + self.obs_horizon]
        
        action_seq = ep['action'][t : t + self.pred_horizon]
        
        return {
            "obs": torch.tensor(state_seq, dtype=torch.float32),
            "action": torch.tensor(action_seq, dtype=torch.float32)
        }