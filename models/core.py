import math

import torch
from torch import nn


class SinusoidalPosEmb(nn.Module):
    """
    Computes sinusoidal positional embeddings for diffusion timesteps.
    
    Maps a scalar diffusion timestep t into a high-dimensional continuous 
    vector representation using interlaced sine and cosine frequencies, allowing 
    the network to interpolate between adjacent noise levels.
    
    Args:
        dim (int): The target dimensionality of the output embedding.
    
    Attributes:
        dim (int): Stored target dimensionality.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        Calculates the positional embeddings for a batch of timesteps.

        Args:
            x (torch.Tensor): A 1D tensor of shape (Batch,) containing the 
                diffusion timesteps.

        Returns:
            torch.Tensor: A 2D tensor of shape (Batch, dim) containing the 
                sinusoidal embeddings.
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ConditionalResidualBlock1D(nn.Module):
    """
    A 1D Convolutional Residual Block with Feature-wise Linear Modulation (FiLM) conditioning.

    Applies a sequence of 1D convolutions, Group Normalization, and Mish activations to a temporal action sequence. 
    The global conditioning vector (combining time and observation features) is 
    projected and added to the hidden state, effectively shifting the feature 
    maps based on the physical state.

    Args:
        in_channels (int): Number of input channels (feature dimensions).
        out_channels (int): Number of output channels.
        cond_dim (int): The dimensionality of the global conditioning vector.

    Attributes:
        blocks (nn.Sequential): The convolutional layers and activations.
        cond_encoder (nn.Sequential): MLP projecting the condition to match `out_channels`.
        residual_conv (nn.Module): 1x1 Conv or Identity mapping for the skip connection.
    """
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.GroupNorm(8, out_channels),
            nn.Mish(),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.GroupNorm(8, out_channels),
            nn.Mish()
        )
        
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels)
        )
        
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        """
        Forward pass applying the block operations and FiLM injection.

        Args:
            x (torch.Tensor): The input action trajectory tensor of shape 
                (Batch, in_channels, Prediction_Horizon).
            cond (torch.Tensor): The global conditioning tensor of shape 
                (Batch, cond_dim).

        Returns:
            torch.Tensor: The output tensor of shape (Batch, out_channels, Prediction_Horizon).
        """
        out = self.blocks[0](x)
        out = self.blocks[1](out)
        out = self.blocks[2](out)
        
        cond_embed = self.cond_encoder(cond).unsqueeze(-1) 
        out = out + cond_embed
        
        out = self.blocks[3](out)
        out = self.blocks[4](out)
        out = self.blocks[5](out)
        
        return out + self.residual_conv(x)

class ConditionalUnet1D(nn.Module):
    """
    A 1D Temporal U-Net architecture predicting the noise epsilon added to 
    an action trajectory during the diffusion process.

    This architecture flattens the past observation sequence and concatenates 
    it with the timestep embedding to form a global conditioning vector. 
    This vector modulates the 1D convolutions at every down-sampling and 
    up-sampling stage via FiLM.

    Args:
        action_dim (int, optional): Dimensionality of the target action space. Defaults to 2.
        obs_dim (int, optional): Dimensionality of the physical state space. Defaults to 5.
        obs_horizon (int, optional): Number of past observation steps (T_o). Defaults to 2.
        global_cond_dim (int, optional): Dimensionality of the intermediate time/obs embeddings. Defaults to 128.
    """
    def __init__(self, action_dim=2, obs_dim=5, obs_horizon=2, global_cond_dim=128):
        super().__init__()
        
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.obs_dim = obs_dim
        
        self.time_emb = SinusoidalPosEmb(global_cond_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(global_cond_dim, global_cond_dim * 4),
            nn.Mish(),
            nn.Linear(global_cond_dim * 4, global_cond_dim)
        )
        
        self.obs_mlp = nn.Sequential(
            nn.Linear(obs_horizon * obs_dim, global_cond_dim),
            nn.Mish(),
            nn.Linear(global_cond_dim, global_cond_dim)
        )
        
        cond_dim = global_cond_dim * 2
        
        down_channels = [action_dim, 64, 128, 256]
        
        self.down1 = ConditionalResidualBlock1D(down_channels[0], down_channels[1], cond_dim)
        self.down2 = ConditionalResidualBlock1D(down_channels[1], down_channels[2], cond_dim)
        self.down3 = ConditionalResidualBlock1D(down_channels[2], down_channels[3], cond_dim)
        
        self.mid = ConditionalResidualBlock1D(down_channels[3], down_channels[3], cond_dim)
        
        self.up1 = ConditionalResidualBlock1D(down_channels[3] * 2, down_channels[2], cond_dim)
        self.up2 = ConditionalResidualBlock1D(down_channels[2] * 2, down_channels[1], cond_dim)
        self.up3 = ConditionalResidualBlock1D(down_channels[1] * 2, down_channels[1], cond_dim)
        
        self.final_conv = nn.Conv1d(down_channels[1], action_dim, kernel_size=1)

    def forward(self, sample, timestep, obs_cond):
        """
        Forward pass to predict the added Gaussian noise.

        Args:
            sample (torch.Tensor): The noisy action trajectory X_k of shape 
                (Batch, Prediction_Horizon, action_dim).
            timestep (torch.Tensor): A 1D tensor of scalar diffusion steps 
                of shape (Batch,).
            obs_cond (torch.Tensor): The physical observation history O_t 
                of shape (Batch, obs_horizon, obs_dim).

        Returns:
            torch.Tensor: The predicted noise epsilon_theta of identical 
                shape to `sample` (Batch, Prediction_Horizon, action_dim).
        """
        t_emb = self.time_mlp(self.time_emb(timestep))
        
        obs_cond = obs_cond.flatten(start_dim=1)
        obs_emb = self.obs_mlp(obs_cond)
        
        global_cond = torch.cat([t_emb, obs_emb], dim=-1)
        
        x = sample.transpose(1, 2)
        
        h1 = self.down1(x, global_cond)
        h2 = self.down2(h1, global_cond)
        h3 = self.down3(h2, global_cond)
        
        m = self.mid(h3, global_cond)
        
        x = self.up1(torch.cat([m, h3], dim=1), global_cond)
        x = self.up2(torch.cat([x, h2], dim=1), global_cond)
        x = self.up3(torch.cat([x, h1], dim=1), global_cond)
        
        out = self.final_conv(x)
        
        return out.transpose(1, 2)