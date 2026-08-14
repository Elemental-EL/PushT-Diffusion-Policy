import json
import math
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import cfg
from dataset.pusht_dataset import PushTStateDataset
from models.core import ConditionalUnet1D
from models.diffusion import DDPMScheduler
from models.ema import EMAModel

"""Diffusion Policy Training Engine.

This module executes the end-to-end training pipeline for a 1D-CNN based 
Conditional Denoising Diffusion Probabilistic Model (DDPM) on low-dimensional 
robot state-action trajectories. It incorporates Exponential Moving Average (EMA) 
weight tracking, dynamic warmup with cosine decay scheduling, and structured 
metric logging.
"""

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer, 
    num_warmup_steps: int, 
    num_training_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Creates a learning rate scheduler with linear warmup and cosine decay.

    Args:
        optimizer (torch.optim.Optimizer): The PyTorch optimizer to schedule.
        num_warmup_steps (int): Total number of linear warmup steps.
        num_training_steps (int): Total number of training steps across all epochs.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: PyTorch learning rate scheduler instance.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def format_duration(seconds: float) -> str:
    """Converts a duration in seconds to a human-readable HH:MM:SS string.

    Args:
        seconds (float): Duration in seconds.

    Returns:
        str: Formatted duration string.
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}h:{minutes:02d}m:{secs:02d}s"


def train() -> None:
    """Executes the full Diffusion Policy training loop.
    
    Loads the normalized PushT state dataset, instantiates the 1D Conditional 
    U-Net and DDPM noise scheduler, and optimizes network parameters using 
    AdamW. Maintains a shadow EMA model across batch iterations.

    Artifacts Exported:
        - checkpoints/best_model_ema.pth: Optimal EMA weights.
        - checkpoints/best_model_raw.pth: Optimal raw network weights.
        - checkpoints/model_epoch_{epoch}_ema.pth: Periodic EMA snapshots.
        - logs/training_metrics.json: Structured history of training statistics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"  DIFFUSION POLICY TRAINING ENGINE | Device: {device}")
    print("=" * 80)

    dataset = PushTStateDataset(
        dataset_path=cfg.dataset_path,
        pred_horizon=cfg.pred_horizon,
        obs_horizon=cfg.obs_horizon,
        action_horizon=cfg.action_horizon
    )
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    total_steps = cfg.num_epochs * len(dataloader)

    unet = ConditionalUnet1D(action_dim=cfg.action_dim, obs_dim=cfg.obs_dim, obs_horizon=cfg.obs_horizon, global_cond_dim=cfg.global_cond_dim).to(device)
    ema = EMAModel(unet, decay=cfg.ema_decay).to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=100).to(device)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=cfg.warmup_steps, 
        num_training_steps=total_steps
    )
    loss_fn = nn.MSELoss()

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    metrics_history = {
        'epoch': [],
        'loss': [],
        'lr': [],
        'epoch_duration': []
    }

    unet.train()
    best_loss = float('inf')
    start_total_time = time.time()

    print(f"Total Epochs: {cfg.num_epochs} | Total Iterations: {total_steps} | Batch Size: {cfg.batch_size}")
    print("-" * 80)

    for epoch in range(cfg.num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0

        for batch in dataloader:
            obs = batch['obs'].to(device)
            action = batch['action'].to(device)

            noise = torch.randn_like(action, device=device)
            bsz = action.shape[0]
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=device).long()

            noisy_actions = noise_scheduler.add_noise(action, noise, timesteps)
            noise_pred = unet(noisy_actions, timesteps, obs)

            loss = loss_fn(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            ema.update(unet)

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        epoch_duration = time.time() - epoch_start_time
        current_lr = lr_scheduler.get_last_lr()[0]

        metrics_history['epoch'].append(epoch)
        metrics_history['loss'].append(avg_loss)
        metrics_history['lr'].append(current_lr)
        metrics_history['epoch_duration'].append(epoch_duration)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(unet.state_dict(), f'{cfg.ckpt_dir}/best_model_raw.pth')
            torch.save(ema.model.state_dict(), f'{cfg.ckpt_dir}/best_model_ema.pth')

        if epoch % 100 == 0 and epoch > 0:
            torch.save(ema.model.state_dict(), f'{cfg.ckpt_dir}/model_epoch_{epoch}_ema.pth')

        if epoch % 50 == 0 or epoch == cfg.num_epochs - 1:
            elapsed = time.time() - start_total_time
            remaining = (elapsed / (epoch + 1)) * (cfg.num_epochs - (epoch + 1))
            print(
                f"Epoch [{epoch:04d}/{cfg.num_epochs:04d}] | "
                f"Loss: {avg_loss:.6f} | "
                f"Best: {best_loss:.6f} | "
                f"LR: {current_lr:.4e} | "
                f"Time/Ep: {epoch_duration:.2f}s | "
                f"Elapsed: {format_duration(elapsed)} | "
                f"ETA: {format_duration(remaining)}"
            )

    with open(f'{cfg.log_dir}/training_metrics.json', 'w') as f:
        json.dump(metrics_history, f, indent=4)

    total_training_time = time.time() - start_total_time
    print("=" * 80)
    print(f"Training Complete. Total Runtime: {format_duration(total_training_time)} | Best Loss: {best_loss:.6f}")
    print("Artifacts saved: 'checkpoints/best_model_ema.pth', 'logs/training_metrics.json'")
    print("=" * 80)


if __name__ == '__main__':
    train()