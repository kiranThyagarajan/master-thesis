from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


root = Path(__file__).resolve().parents[1]
train_dir = root / "data_pendulum" / "train"
test_dir = root / "data_pendulum" / "test"


def load_dataset_split(split_dir):
    """
    Load all trajectories from one dataset split.

    Each trajectory directory is expected to contain:
        observations.npy: shape (12, 100)
        controls.npy:     shape (1, 99)

    Returns:
        trajectories:
            shape (num_trajectories, 100, 12)

        controls:
            shape (num_trajectories, 99, 1)
    """
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {split_dir}"
        )

    all_trajectories = []
    all_controls = []

    for traj_dir in sorted(split_dir.iterdir()):
        if not traj_dir.is_dir():
            continue

        observations_path = traj_dir / "observations.npy"
        controls_path = traj_dir / "controls.npy"

        if not observations_path.exists():
            print(
                f"Skipping {traj_dir.name}: "
                f"observations.npy was not found"
            )
            continue

        if not controls_path.exists():
            print(
                f"Skipping {traj_dir.name}: "
                f"controls.npy was not found"
            )
            continue

        observations = np.load(observations_path)
        controls = np.load(controls_path)

        # Expected saved shapes:
        # observations: (12, 100)
        # controls:     (1, 99)
        if observations.ndim != 2:
            raise ValueError(
                f"{observations_path} must be a 2D array, "
                f"but received shape {observations.shape}"
            )

        if controls.ndim != 2:
            raise ValueError(
                f"{controls_path} must be a 2D array, "
                f"but received shape {controls.shape}"
            )

        if observations.shape[0] != 12:
            raise ValueError(
                f"Expected 12 observation dimensions in "
                f"{observations_path}, but received "
                f"shape {observations.shape}"
            )

        if controls.shape[0] != 1:
            raise ValueError(
                f"Expected 1 control dimension in "
                f"{controls_path}, but received "
                f"shape {controls.shape}"
            )

        if controls.shape[1] != observations.shape[1] - 1:
            raise ValueError(
                f"Time dimension mismatch in {traj_dir.name}: "
                f"observations have {observations.shape[1]} steps, "
                f"but controls have {controls.shape[1]} steps"
            )

        # Convert from:
        # observations: (12, 100) -> (100, 12)
        # controls:     (1, 99)   -> (99, 1)
        all_trajectories.append(
            observations.T.astype(np.float64)
        )

        all_controls.append(
            controls.T.astype(np.float64)
        )

    if not all_trajectories:
        raise RuntimeError(
            f"No valid trajectories were found in {split_dir}"
        )

    trajectories = np.stack(
        all_trajectories,
        axis=0,
    )

    controls = np.stack(
        all_controls,
        axis=0,
    )

    return trajectories, controls


# ---------------------------------------------------------
# Load training dataset
# ---------------------------------------------------------

trajectories_train, controls_train = load_dataset_split(
    train_dir
)

# trajectories_train:
# (num_train_trajectories, 100, 12)

# controls_train:
# (num_train_trajectories, 99, 1)


# ---------------------------------------------------------
# Load testing dataset
# ---------------------------------------------------------

trajectories_test, controls_test = load_dataset_split(
    test_dir
)

# trajectories_test:
# (num_test_trajectories, 100, 12)

# controls_test:
# (num_test_trajectories, 99, 1)


# ---------------------------------------------------------
# Convert NumPy arrays to PyTorch tensors
# ---------------------------------------------------------

trajectories_tensor_train = torch.from_numpy(
    trajectories_train
).to(torch.float64)

controls_tensor_train = torch.from_numpy(
    controls_train
).to(torch.float64)

trajectories_tensor_test = torch.from_numpy(
    trajectories_test
).to(torch.float64)

controls_tensor_test = torch.from_numpy(
    controls_test
).to(torch.float64)


# ---------------------------------------------------------
# Print dataset shapes
# ---------------------------------------------------------

print(
    "Training trajectories shape:",
    trajectories_tensor_train.shape,
)
# (num_train_trajectories, 100, 12)

print(
    "Training controls shape:",
    controls_tensor_train.shape,
)
# (num_train_trajectories, 99, 1)

print(
    "Testing trajectories shape:",
    trajectories_tensor_test.shape,
)
# (num_test_trajectories, 100, 12)

print(
    "Testing controls shape:",
    controls_tensor_test.shape,
)
# (num_test_trajectories, 99, 1)


# ---------------------------------------------------------
# Flatten and plot all training control values
# ---------------------------------------------------------

train_controls_flat = controls_train.reshape(
    -1,
    controls_train.shape[-1],
)

# Shape:
# (num_train_trajectories * 99, 1)

plt.figure(figsize=(12, 4))

for control_index in range(train_controls_flat.shape[1]):
    plt.plot(
        train_controls_flat[:, control_index],
        ".",
        label=f"Control dimension {control_index}",
        alpha=0.7,
    )

plt.xlabel("Sample index")
plt.ylabel("Torque")
plt.title("All single-pendulum training control points")
plt.legend()
plt.tight_layout()
plt.show()