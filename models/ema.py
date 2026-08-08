import copy

import torch
from torch import nn


class EMAModel:
    """Maintains an Exponential Moving Average (EMA) of model parameters.

    Creates an evaluated shadow copy of the model whose weights are updated
    using a decayed moving average after each optimization step.

    Args:
        model (nn.Module): The source neural network to track.
        decay (float): Maximum exponential decay rate. Defaults to 0.9999.
        min_decay (float): Minimum decay floor during warmup. Defaults to 0.0.
        update_after_step (int): Optimization step at which EMA updates begin.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 0
    ) -> None:
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.model.requires_grad_(False)
        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.step_count = 0

    def get_decay(self) -> float:
        """Computes the dynamic decay rate based on optimization step count.

        Returns:
            float: The current step-adjusted decay value.
        """
        if self.step_count < self.update_after_step:
            return 0.0
        step = self.step_count - self.update_after_step
        value = (1 + step) / (10 + step)
        return max(self.min_decay, min(self.decay, value))

    def update(self, model: nn.Module) -> None:
        """Updates shadow parameters using the current source model weights.

        Args:
            model (nn.Module): The optimized source model.
        """
        self.step_count += 1
        decay = self.get_decay()

        with torch.no_grad():
            for ema_param, param in zip(self.model.parameters(), model.parameters()):
                ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)
            for ema_buffer, buffer in zip(self.model.buffers(), model.buffers()):
                ema_buffer.data.copy_(buffer.data)

    def to(self, device: torch.device) -> 'EMAModel':
        """Transfers the shadow model to the specified target device.

        Args:
            device (torch.device): PyTorch execution device.

        Returns:
            EMAModel: The instance moved to the target device.
        """
        self.model.to(device)
        return self