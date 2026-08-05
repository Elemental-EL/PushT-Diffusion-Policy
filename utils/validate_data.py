import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset.normalization import unnormalize_data
from dataset.pusht_dataset import PushTStateDataset


def validate():
    """
    Validates the spatial and temporal integrity of the PushT data pipeline.

    Fetches a single batch from the PushTStateDataset, reverses the Min-Max normalization, and plots the physical coordinates of the agent and block to verify action chunking alignment. 

    The PushT low-dimensional state vector is assumed to be 5D:
    [agent_x, agent_y, block_x, block_y, block_angle].

    Validation Criteria:
        - Spatial Continuity: The start of the predicted action trajectory (red) 
          must perfectly align with the agent's current position (end of the green line).
        - Boundary Constraints: Un-normalized coordinates must fall within the 
          [0, 512] pixel boundary of the PushT environment.
    """
    dataset = PushTStateDataset(
        dataset_path='data/pusht_cchi_v7_replay.zarr',
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=8
    )
    
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    batch = next(iter(dataloader))
    
    obs_seq_norm = batch['obs'][0].numpy()
    action_seq_norm = batch['action'][0].numpy()
    
    obs_seq = unnormalize_data(obs_seq_norm, dataset.stats["state"])
    action_seq = unnormalize_data(action_seq_norm, dataset.stats["action"])
    
    plt.figure(figsize=(6, 6))
    
    block_x = obs_seq[-1, 2]
    block_y = obs_seq[-1, 3]
    plt.plot(block_x, block_y, 'bs', markersize=15, label="Block")
    
    agent_past_x = obs_seq[:, 0]
    agent_past_y = obs_seq[:, 1]
    plt.plot(agent_past_x, agent_past_y, 'go-', markersize=6, label="Past Trajectory")
    
    agent_future_x = action_seq[:, 0]
    agent_future_y = action_seq[:, 1]
    plt.plot(agent_future_x, agent_future_y, 'rx-', markersize=6, label="Future Actions")
    
    plt.xlim(0, 512)
    plt.ylim(0, 512)
    plt.gca().invert_yaxis()
    plt.title("Data Pipeline Validation")
    plt.legend(["Block", "Past Trajectory", "Future Actions"])
    plt.show()

if __name__ == "__main__":
    validate()