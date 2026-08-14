from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import cfg
from dataset.normalization import normalize_data, unnormalize_data
from dataset.pusht_dataset import PushTStateDataset
from env.pusht_wrapper import get_pusht_env
from models.core import ConditionalUnet1D
from models.diffusion import DDPMScheduler

"""Diffusion Policy Evaluation and Receding Horizon Control (RHC).

This module evaluates a trained DDPM policy within the PushT Gym physics 
environment across multiple random initializations. It implements 
Receding Horizon Control (action chunking), executing only the first T_a actions 
of a predicted T_p trajectory before re-planning.
"""

def evaluate() -> None:
    """Evaluates the trained Diffusion Policy using Receding Horizon Control (RHC).
    
    Loads model weights from the EMA checkpoint, queries the PushT physics 
    environment, runs reverse DDPM denoising at each planning step, and executes 
    receding horizon actions. Computes both maximum and terminal target coverage 
    (IoU) metrics and visualizes 2D agent-block trajectories.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on: {device}")

    seeds = [42, 100, 2026, 777, 999]

    print("Extracting normalization statistics from dataset...")
    dataset = PushTStateDataset(cfg.dataset_path, cfg.pred_horizon, cfg.obs_horizon, cfg.action_horizon)
    stats = dataset.stats

    unet = ConditionalUnet1D(action_dim=cfg.action_dim, obs_dim=cfg.obs_dim, obs_horizon=cfg.obs_horizon, global_cond_dim=cfg.global_cond_dim).to(device)
    unet.load_state_dict(torch.load(f'{cfg.ckpt_dir}/best_model_ema.pth', map_location=device))
    unet.eval()
    
    noise_scheduler = DDPMScheduler(num_train_timesteps=100).to(device)
    env = get_pusht_env()

    _, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for seed_idx, seed in enumerate(seeds):
        env.seed(seed)
        obs = env.reset()
        
        obs_deque = deque([obs] * cfg.obs_horizon, maxlen=cfg.obs_horizon)
        
        agent_path = []
        block_path = []
        
        done = False
        step_idx = 0
        max_reward = 0.0
        final_reward = 0.0

        while not done and step_idx < cfg.max_env_steps:
            obs_seq = np.stack(obs_deque)
            nobs = normalize_data(obs_seq, stats['state'])
            nobs_tensor = torch.tensor(nobs, dtype=torch.float32, device=device).unsqueeze(0)
            
            naction = torch.randn((1, cfg.pred_horizon, 2), device=device)
            for k in reversed(range(100)):
                k_tensor = torch.tensor([k], device=device).long()
                with torch.no_grad():
                    noise_pred = unet(naction, k_tensor, nobs_tensor)
                naction = noise_scheduler.step(noise_pred, k, naction)
            
            naction = naction.squeeze(0).cpu().numpy()
            action_pred = unnormalize_data(naction, stats['action'])
            
            for i in range(cfg.action_horizon):
                obs, reward, done, _ = env.step(action_pred[i])
                
                obs_deque.append(obs)
                agent_path.append((obs[0], obs[1]))
                block_path.append((obs[2], obs[3]))
                
                max_reward = max(max_reward, reward)
                final_reward = reward
                step_idx += 1
                
                if done:
                    break

        print(f"Seed {seed:4d} | Steps: {step_idx:3d} | Max Coverage: {max_reward:.3f} | Final Coverage: {final_reward:.3f}")
        
        ax = axes[seed_idx]
        agent_path = np.array(agent_path)
        block_path = np.array(block_path)
        
        ax.plot(agent_path[:, 0], agent_path[:, 1], 'g-', label='Agent', alpha=0.6)
        ax.plot(block_path[:, 0], block_path[:, 1], 'b-', label='Block', linewidth=2)
        ax.plot(256, 256, 'r*', markersize=15, label='Target (256, 256)') 
        
        ax.set_xlim(0, 512)
        ax.set_ylim(0, 512)
        ax.invert_yaxis()
        ax.set_title(f"Seed {seed}\nMax Cov: {max_reward:.2f}")
        if seed_idx == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=300)
    print("\n[SUCCESS] Evaluation complete. Saved to 'evaluation_results.png'.")


if __name__ == '__main__':
    evaluate()