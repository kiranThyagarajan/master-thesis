import cv2
import numpy as np


def split_dataset_by_trajectory(dataset, train_fraction=0.7, val_fraction=0.15):
    test_fraction = 1.0 - train_fraction - val_fraction
    if test_fraction < 0:
        raise ValueError("train_fraction + val_fraction must be <= 1.0")

    n_traj = len(dataset)
    if n_traj < 3:
        raise ValueError("Need at least 3 trajectories for train/val/test split.")

    n_train = max(1, int(np.floor(train_fraction * n_traj)))
    n_val = max(1, int(np.floor(val_fraction * n_traj)))
    n_test = n_traj - n_train - n_val

    if n_test < 1:
        if n_val > 1:
            n_val -= 1
        else:
            n_train -= 1

    ordered_ids = np.arange(n_traj, dtype=np.int32)
    train_ids = ordered_ids[:n_train]
    val_ids = ordered_ids[n_train:n_train + n_val]
    test_ids = ordered_ids[n_train + n_val:]

    train_set = [(int(traj_id), dataset[int(traj_id)]) for traj_id in train_ids]
    val_set = [(int(traj_id), dataset[int(traj_id)]) for traj_id in val_ids]
    test_set = [(int(traj_id), dataset[int(traj_id)]) for traj_id in test_ids]

    return train_set, val_set, test_set, train_ids, val_ids, test_ids


def save_trajectory_dataset(split_dataset, split_dir):
    split_dir.mkdir(parents=True, exist_ok=True)

    for traj_id, traj in split_dataset:
        traj_dir = split_dir / f"traj_{traj_id + 1:04d}"
        frames_dir = traj_dir / "frames"
        traj_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)

        for frame_idx, image in enumerate(traj["I"]):
            cv2.imwrite(str(frames_dir / f"{frame_idx:04d}.png"), image)

        np.save(traj_dir / "controls.npy", traj["U"][:-1].T)  # shape (2, N-1)
        #np.save(traj_dir / "markers.npy", traj["Y"])
        np.save(traj_dir / "observations.npy", traj["O"].T)  # shape (24, N)


def print_trajectory_split_summary(split_dataset, name="Split"):
    num_trajectories = len(split_dataset)
    num_steps = sum(len(traj["O"]) for _, traj in split_dataset)
    num_frames = sum(len(traj["I"]) for _, traj in split_dataset)

    print(f"\n--- {name} Summary ---")
    print("Trajectories:", num_trajectories)
    print("Steps       :", num_steps)
    print("Frames      :", num_frames)
