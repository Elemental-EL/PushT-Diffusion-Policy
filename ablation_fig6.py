"""Diffusion Policy Ablation Study: Figure 6.

This module automates the simulation rollouts required to reproduce Figure 6 
from the Diffusion Policy paper. It evaluates two critical trade-offs:
1. Action Horizon (T_a): The balance between temporal consistency and responsiveness.
2. Latency Robustness: The policy's ability to maintain high success rates when 
   observations are delayed by L steps.
"""

import json
import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset.normalization import normalize_data, unnormalize_data
from dataset.pusht_dataset import PushTStateDataset
from env.pusht_wrapper import get_pusht_env
from models.core import ConditionalUnet1D
from models.diffusion import DDPMScheduler


def evaluate_policy(unet, noise_scheduler, stats, env, device, ta: int, latency: int, test_seeds: list) -> float:
    """Evaluates the policy under specific action horizon and latency constraints.

    Args:
        unet (nn.Module): The trained Conditional 1D U-Net.
        noise_scheduler (DDPMScheduler): The DDPM noise scheduler.
        stats (dict): Dataset normalization statistics.
        env (gym.Env): The PushT physics environment.
        device (torch.device): Compute device.
        ta (int): Action horizon (number of steps to execute before replanning).
        latency (int): Observation delay in steps.
        test_seeds (list): List of random seeds for environment initialization.

    Returns:
        float: The mean success rate (coverage >= 0.95) across all test seeds.
    """
    pred_horizon = 128
    obs_horizon = 2
    max_steps = 300
    
    buffer_maxlen = obs_horizon + latency
    
    success_count = 0

    for seed in test_seeds:
        env.seed(seed)
        obs = env.reset()
        
        obs_buffer = deque([obs] * buffer_maxlen, maxlen=buffer_maxlen)
        
        step_idx = 0
        max_coverage = 0.0
        done = False

        while not done and step_idx < max_steps:
            if latency > 0:
                obs_seq = np.stack(list(obs_buffer)[:obs_horizon])
            else:
                obs_seq = np.stack(list(obs_buffer))
                
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

            for i in range(ta):
                obs, reward, done, _ = env.step(action_pred[i])
                obs_buffer.append(obs)
                max_coverage = max(max_coverage, reward)
                step_idx += 1

                if done:
                    break

        if max_coverage >= 0.95:
            success_count += 1
            
    return success_count / len(test_seeds)


def run_fig6_ablations() -> None:
    """Executes the dual ablation study and plots the Relative Performance Change."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"  DIFFUSION POLICY ABLATION SUITE (FIGURE 6) | Device: {device}")
    print("=" * 80)

    num_seeds = 20
    test_seeds = list(range(2000, 2000 + num_seeds))
    
    print("Loading dataset normalization statistics...")
    dataset = PushTStateDataset('data/pusht_cchi_v7_replay.zarr', 128, 2, 8)
    stats = dataset.stats

    unet = ConditionalUnet1D().to(device)
    unet.load_state_dict(torch.load('checkpoints/best_model_ema.pth', map_location=device))
    unet.eval()
    noise_scheduler = DDPMScheduler(num_train_timesteps=100).to(device)
    env = get_pusht_env()

    results = {'action_horizon': {}, 'latency': {}}

    action_horizons = [1, 2, 4, 8, 16, 32, 64, 128]
    print("\n[Phase 1] Evaluating Action Horizon Trade-off...")
    for ta in action_horizons:
        sr = evaluate_policy(unet, noise_scheduler, stats, env, device, ta, latency=0, test_seeds=test_seeds)
        results['action_horizon'][ta] = sr
        print(f"  T_a = {ta:2d} | Success Rate: {sr * 100:.1f}%")

    latencies = [0, 1, 2, 3, 4, 5, 6, 7]
    print("\n[Phase 2] Evaluating Latency Robustness (T_a = 8)...")
    for l in latencies:
        sr = evaluate_policy(unet, noise_scheduler, stats, env, device, ta=8, latency=l, test_seeds=test_seeds)
        results['latency'][l] = sr
        print(f"  Latency = {l:d} steps | Success Rate: {sr * 100:.1f}%")

    os.makedirs('logs', exist_ok=True)
    with open('logs/ablation_fig6.json', 'w') as f:
        json.dump(results, f, indent=4)

    ta_keys = list(results['action_horizon'].keys())
    ta_vals = np.array([results['action_horizon'][k] for k in ta_keys])
    ta_rel_change = ta_vals - np.max(ta_vals)

    lat_keys = list(results['latency'].keys())
    lat_vals = np.array([results['latency'][k] for k in lat_keys])
    lat_rel_change = lat_vals - np.max(lat_vals)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    ax1.plot(ta_keys, ta_rel_change, marker='o', linewidth=2, color='tab:blue', label='PushT')
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(ta_keys)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.set_xlabel('Action Horizon (steps)', fontsize=12)
    ax1.set_ylabel('Relative Perf Change', fontsize=12)
    ax1.set_title('Action Horizon', fontsize=14)
    ax1.grid(True, linestyle='-', alpha=0.7)
    ax1.set_ylim(-0.65, 0.05)

    ax2.plot(lat_keys, lat_rel_change, marker='o', linewidth=2, color='tab:blue', label='PushT')
    ax2.set_xticks(lat_keys)
    ax2.set_xlabel('Latency (steps)', fontsize=12)
    ax2.set_title('Latency Robustness', fontsize=14)
    ax2.grid(True, linestyle='-', alpha=0.7)

    lines, labels = ax1.get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=12)

    plt.tight_layout()
    plt.savefig('ablation_figure_6.png', dpi=300, bbox_inches='tight')
    print("\n[SUCCESS] Figure 6 generated and saved to 'ablation_figure_6.png'.")


if __name__ == '__main__':
    run_fig6_ablations()