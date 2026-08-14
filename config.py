"""Centralized Hyperparameter Configuration for Diffusion Policy."""

from dataclasses import dataclass


@dataclass
class PushTConfig:
    # Model & Dimensions
    action_dim: int = 2
    obs_dim: int = 5
    global_cond_dim: int = 128

    # Receding Horizon Constraints
    pred_horizon: int = 128
    obs_horizon: int = 2
    action_horizon: int = 8
    
    # Environment Constraints
    max_env_steps: int = 300
    
    # Training Hyperparameters
    num_epochs: int = 5000
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 1e-6
    warmup_steps: int = 500
    ema_decay: float = 0.9999
    
    # File Paths
    dataset_path: str = 'data/pusht_cchi_v7_replay.zarr'
    ckpt_dir: str = 'checkpoints'
    log_dir: str = 'logs'

# Instantiate a global configuration object
cfg = PushTConfig()