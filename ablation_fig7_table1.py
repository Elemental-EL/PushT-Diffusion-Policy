"""Diffusion Policy Training Stability (Fig 7) and Benchmark (Table I).

This script chronologically evaluates saved EMA checkpoints to map the learning 
progression of the policy. It evaluates across 50 random initial conditions to 
satisfy the strict variance requirements of Table I.
"""

import glob
import json
import os
import re
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset.normalization import normalize_data, unnormalize_data
from dataset.pusht_dataset import PushTStateDataset
from env.pusht_wrapper import get_pusht_env
from models.core import ConditionalUnet1D
from models.diffusion import DDPMScheduler


def evaluate_checkpoint(
    ckpt_path: str, 
    unet: torch.nn.Module, 
    noise_scheduler: DDPMScheduler, 
    stats: dict, 
    env, 
    device: torch.device, 
    test_seeds: list
) -> float:
    """Runs closed-loop inference for a specific model checkpoint.
    
    Args:
        ckpt_path: Path to the .pth weights file.
        unet: Initialized 1D-CNN U-Net.
        noise_scheduler: DDPM Scheduler.
        stats: Dataset normalization boundaries.
        env: PushT Environment.
        device: Execution hardware.
        test_seeds: List of 50 environment seeds.
        
    Returns:
        float: Success rate (episodes achieving >= 0.95 IoU).
    """
    unet.load_state_dict(torch.load(ckpt_path, map_location=device))
    unet.eval()

    pred_horizon = 128
    obs_horizon = 2
    action_horizon = 8
    max_steps = 300
    
    success_count = 0

    for seed in test_seeds:
        env.seed(seed)
        obs = env.reset()
        obs_deque = deque([obs] * obs_horizon, maxlen=obs_horizon)
        
        step_idx = 0
        max_coverage = 0.0
        done = False

        while not done and step_idx < max_steps:
            obs_seq = np.stack(obs_deque)
            nobs = normalize_data(obs_seq, stats['state'])
            nobs_tensor = torch.tensor(nobs, dtype=torch.float32, device=device).unsqueeze(0)

            naction = torch.randn((1, pred_horizon, 2), device=device)
            for k in reversed(range(100)):
                k_tensor = torch.tensor([k], device=device).long()
                with torch.no_grad():
                    noise_pred = unet(naction, k_tensor, nobs_tensor)
                naction = noise_scheduler.step(noise_pred, k, naction)

            naction = naction.squeeze(0).cpu().numpy()
            action_pred = unnormalize_data(naction, stats['action'])

            for i in range(action_horizon):
                obs, reward, done, _ = env.step(action_pred[i])
                obs_deque.append(obs)
                max_coverage = max(max_coverage, reward)
                step_idx += 1
                if done:
                    break

        if max_coverage >= 0.95:
            success_count += 1
            
    return success_count / len(test_seeds)


def run_training_stability_analysis() -> None:
    """Executes the checkpoint sweep and generates Fig 7 / Table I deliverables."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"  TRAINING STABILITY (FIG 7) & BENCHMARK (TABLE I) | Device: {device}")
    print("=" * 80)

    test_seeds = list(range(3000, 3050))
    
    print("Extracting normalization statistics...")
    dataset = PushTStateDataset('data/pusht_cchi_v7_replay.zarr', 128, 2, 8)
    stats = dataset.stats

    unet = ConditionalUnet1D().to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=100).to(device)
    env = get_pusht_env()

    ckpt_files = glob.glob('checkpoints/model_epoch_*_ema.pth')
    
    epoch_to_ckpt = {}
    for f in ckpt_files:
        match = re.search(r'model_epoch_(\d+)_ema\.pth', f)
        if match:
            epoch = int(match.group(1))
            epoch_to_ckpt[epoch] = f

    if not epoch_to_ckpt:
        print("[ERROR] No intermediate checkpoints found in 'checkpoints/'.")
        return

    early_epochs = [200, 400, 600, 800, 1000]
    mid_epochs = [1500, 2000, 2500, 3000, 3500]
    
    all_epochs_sorted = sorted(epoch_to_ckpt.keys())
    
    last_10_epochs = all_epochs_sorted[-10:] if len(all_epochs_sorted) >= 10 else all_epochs_sorted
    
    target_epochs = sorted(set(early_epochs + mid_epochs + last_10_epochs))
    
    sorted_epochs = [e for e in target_epochs if e in epoch_to_ckpt]

    results = {}
    
    print(f"Evaluating {len(sorted_epochs)} targeted checkpoints across 50 seeds each...")
    for epoch in sorted_epochs:
        ckpt = epoch_to_ckpt[epoch]
        sr = evaluate_checkpoint(ckpt, unet, noise_scheduler, stats, env, device, test_seeds)
        results[epoch] = sr
        print(f"  Epoch {epoch:04d} | Success Rate: {sr * 100:.1f}%")

    os.makedirs('logs', exist_ok=True)
    with open('logs/training_stability.json', 'w') as f:
        json.dump(results, f, indent=4)

    success_rates = list(results.values())
    max_performance = max(success_rates)
    
    last_10_success_rates = [results[e] for e in last_10_epochs if e in results]
    avg_performance = sum(last_10_success_rates) / len(last_10_success_rates)
    
    print("\n" + "=" * 80)
    print("  DELIVERABLE: TABLE I METRICS (DiffusionPolicy-C, Sim PushT State)")
    print("=" * 80)
    print("  Format: (Max Performance) / (Average of Last 10 Checkpoints)")
    print(f"  Result: {max_performance:.2f} / {avg_performance:.2f}")
    print("=" * 80)

    epochs_list = list(results.keys())
    sr_list = [results[e] for e in epochs_list]

    plt.figure(figsize=(6, 5))
    plt.plot(epochs_list, sr_list, color='tab:blue', linewidth=2, label='Diffusion Policy')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.title('Sim PushT State (Training Stability)', fontsize=14)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle='-', alpha=0.7)
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig('training_stability_fig7.png', dpi=300)
    print("\n[SUCCESS] Figure 7 generated and saved to 'training_stability_fig7.png'.")


if __name__ == '__main__':
    run_training_stability_analysis()