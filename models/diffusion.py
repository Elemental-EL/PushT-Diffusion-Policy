import torch
from torch import nn


class DDPMScheduler(nn.Module):
    """
    Denoising Diffusion Probabilistic Model (DDPM) noise scheduler.

    Implements a linear variance schedule to control the forward noise injection 
    and the reverse Langevin dynamics process. Variables such as alpha 
    and beta are pre-computed and stored as PyTorch buffers to ensure 
    automatic hardware synchronization during training.

    Args:
        num_train_timesteps (int, optional): Total number of diffusion steps (K). Defaults to 100.
        beta_start (float, optional): The initial variance beta_1. Defaults to 0.0001.
        beta_end (float, optional): The final variance beta_K. Defaults to 0.02.
    """
    def __init__(self, num_train_timesteps=100, beta_start=0.0001, beta_end=0.02):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    def add_noise(self, original_samples, noise, timesteps):
        """
        Executes the closed-form forward diffusion process to jump directly to X_k.

        Applies the mathematical formulation:
        X_k = sqrt(alpha_k_bar)*X_0 + sqrt(1 - alpha_k_bar)*epsilon

        Args:
            original_samples (torch.Tensor): The clean action trajectory X_0 
                of shape (Batch, Prediction_Horizon, Action_Dim).
            noise (torch.Tensor): Sampled Gaussian noise epsilon from N(0, I)
                of identical shape to `original_samples`.
            timesteps (torch.Tensor): The diffusion step k for each batch 
                item of shape (Batch,).

        Returns:
            torch.Tensor: The corrupted action trajectory X_k.
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1).unsqueeze(-1)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1).unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def step(self, model_output, timestep, sample):
        """
        Executes a single step of the reverse denoising process.

        Estimates the clean sample X_0 from the current noisy sample X_k and 
        the network's noise prediction, then computes the posterior mean X_k-1. 
        Adds scaled random noise (variance) to mimic Stochastic Langevin Dynamics, 
        unless it is the final step (t=0).

        Args:
            model_output (torch.Tensor): The noise predicted by the U-Net epsilon_theta.
            timestep (int): The current scalar diffusion step t.
            sample (torch.Tensor): The current noisy action trajectory X_k.

        Returns:
            torch.Tensor: The slightly less noisy trajectory X_k-1.
        """
        t = timestep
        
        alpha_t = 1.0 - self.betas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]
        
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        pred_original_sample = (sample - sqrt_one_minus_alpha_cumprod_t * model_output) / torch.sqrt(alpha_cumprod_t)
        
        pred_prev_sample = (torch.sqrt(alpha_t) * (1 - self.alphas_cumprod[t-1]) * sample + 
                            beta_t * torch.sqrt(self.alphas_cumprod[t-1]) * pred_original_sample) / (1 - alpha_cumprod_t) if t > 0 else pred_original_sample

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (1 - self.alphas_cumprod[t-1]) / (1 - alpha_cumprod_t) * beta_t
            variance = torch.sqrt(variance) * noise
            
        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample