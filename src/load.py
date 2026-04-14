from pathlib import Path
import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
train_dir = root / "data" / "train"
test_dir = root / "data" / "test"

all_trajectories_train = []
all_controls_train = []

for traj_dir in sorted(train_dir.iterdir()):
    if not traj_dir.is_dir():
        continue

    markers_path = traj_dir / "observations.npy"
    controls_path = traj_dir / "controls.npy"

    if not markers_path.exists():
        continue

    markers = np.load(markers_path)    # (24, 100)
    controls = np.load(controls_path)  # (2, 99)

    all_trajectories_train.append(markers.T.astype(np.float64))  # [(100, 24), (100, 24), ...]
    all_controls_train.append(controls.T.astype(np.float64))     # [(99, 2), (99, 2), ...]

trajectories_train = np.stack(all_trajectories_train, axis=0)    # (num_train_traj, 100, 24)
controls_train = np.stack(all_controls_train, axis=0)            # (num_train_traj, 99, 2)

all_trajectories_test = []
all_controls_test = []

for traj_dir in sorted(test_dir.iterdir()):
    if not traj_dir.is_dir():
        continue

    markers_path = traj_dir / "observations.npy"
    controls_path = traj_dir / "controls.npy"

    if not markers_path.exists():
        continue

    markers = np.load(markers_path)    # (24, 100)
    controls = np.load(controls_path)  # (2, 99)

    all_trajectories_test.append(markers.T.astype(np.float64))   # [(100, 24), (100, 24), ...]
    all_controls_test.append(controls.T.astype(np.float64))      # [(99, 2), (99, 2), ...]

trajectories_test = np.stack(all_trajectories_test, axis=0)      # (num_test_traj, 100, 24)
controls_test = np.stack(all_controls_test, axis=0)              # (num_test_traj, 99, 2)

trajectories_tensor_train = torch.from_numpy(trajectories_train).to(torch.float64)
controls_tensor_train = torch.from_numpy(controls_train).to(torch.float64)

trajectories_tensor_test = torch.from_numpy(trajectories_test).to(torch.float64)
controls_tensor_test = torch.from_numpy(controls_test).to(torch.float64)

print("Training trajectories shape:", trajectories_tensor_train.shape)  # (num_train_traj, 100, 24)
print("Training controls shape:", controls_tensor_train.shape)          # (num_train_traj, 99, 2)
print("Testing trajectories shape:", trajectories_tensor_test.shape)     # (num_test_traj, 100, 24)
print("Testing controls shape:", controls_tensor_test.shape)             # (num_test_traj, 99, 2)