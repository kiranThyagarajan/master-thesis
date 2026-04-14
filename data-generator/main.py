from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from control import pd_control
from dataset_builder import (
    print_trajectory_split_summary,
    save_trajectory_dataset,
    split_dataset_by_trajectory,
)
from dynamics import qddot
from markers import get_marker_positions, get_marker_observation
from params import get_params
from render import render_markers
from sampling import sample_initial_condition

NUM_TRAJECTORIES = 1429
OUTPUT_DIR = Path("data")
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.15
SEED = None
SEQUENCE_LENGTH = 100
CONVERGENCE_Q_TOL = np.deg2rad(0.5)
CONVERGENCE_QDOT_TOL = np.deg2rad(1.0)
CONVERGENCE_STREAK = 5
MAX_ATTEMPTS_FACTOR = 20


def state_derivative(t, x, p):
    q = x[:2]
    qdot = x[2:]
    tau = pd_control(q, qdot, p)
    qdd = qddot(q, qdot, tau, p)
    return np.hstack((qdot, qdd))


def simulate_trajectory(x0, p):
    dt = p["dt"]
    T = p["T"]
    time = np.arange(0.0, T + dt, dt)

    sol = solve_ivp(
        fun=lambda t, x: state_derivative(t, x, p),
        t_span=(0.0, T),
        y0=x0,
        t_eval=time,
        method="RK45",
        rtol=1e-7,
        atol=1e-9,
    )

    if not sol.success:
        return None

    X = sol.y.T
    N = len(sol.t)
    U = np.zeros((N, 2))
    Y = np.zeros((N, 6, 2))
    O = np.zeros((N, 24))
    I = np.zeros((N, 64, 128), dtype=np.uint8)

    for k in range(N):
        q = X[k, :2]
        qdot = X[k, 2:]
        U[k] = pd_control(q, qdot, p)
        Y[k] = get_marker_positions(q, p)
        O[k] = get_marker_observation(q, qdot, p)
        I[k] = render_markers(Y[k], p)

    return {
        "x0": x0,
        "time": sol.t,
        "X": X,
        "U": U,
        "Y": Y,
        "O": O,
        "I": I,
    }


def find_convergence_index(traj, p, q_tol, qdot_tol, streak_length):
    q = traj["X"][:, :2]
    qdot = traj["X"][:, 2:]
    q_error = np.linalg.norm(q - p["q_target"], axis=1)
    qdot_error = np.linalg.norm(qdot - p["qdot_target"], axis=1)
    converged = (q_error <= q_tol) & (qdot_error <= qdot_tol)

    streak = 0
    for idx, is_converged in enumerate(converged):
        if is_converged:
            streak += 1
            if streak >= streak_length:
                return idx - streak_length + 1
        else:
            streak = 0

    return None


def truncate_trajectory(traj, sequence_length):
    return {
        "x0": traj["x0"],
        "time": traj["time"][:sequence_length],
        "X": traj["X"][:sequence_length],
        "U": traj["U"][:sequence_length],
        "Y": traj["Y"][:sequence_length],
        "O": traj["O"][:sequence_length],
        "I": traj["I"][:sequence_length],
    }


def generate_dataset(p, num_trajectories):
    rng = np.random.default_rng(p.get("seed"))
    dataset = []
    attempts = 0
    rejected = 0
    max_attempts = max(num_trajectories * MAX_ATTEMPTS_FACTOR, num_trajectories)

    while len(dataset) < num_trajectories and attempts < max_attempts:
        attempts += 1
        x0 = sample_initial_condition(p, rng)
        traj = simulate_trajectory(x0, p)

        if traj is None:
            print(f"Attempt {attempts}: trajectory simulation failed")
            continue

        if len(traj["time"]) < SEQUENCE_LENGTH:
            rejected += 1
            print(f"Attempt {attempts}: trajectory too short for {SEQUENCE_LENGTH} steps")
            continue

        convergence_idx = find_convergence_index(
            traj,
            p,
            q_tol=CONVERGENCE_Q_TOL,
            qdot_tol=CONVERGENCE_QDOT_TOL,
            streak_length=CONVERGENCE_STREAK,
        )
        if convergence_idx is not None and convergence_idx < SEQUENCE_LENGTH:
            rejected += 1
            print(f"Attempt {attempts}: rejected, converged too early at step {convergence_idx}")
            continue

        dataset.append(truncate_trajectory(traj, SEQUENCE_LENGTH))
        print(f"Accepted trajectory {len(dataset)}/{num_trajectories} on attempt {attempts}")

    if len(dataset) < num_trajectories:
        raise RuntimeError(
            f"Only collected {len(dataset)} valid trajectories after {attempts} attempts."
        )

    print(f"\nRejected trajectories: {rejected}")

    return dataset


def main():
    p = get_params()
    if SEED is not None:
        p["seed"] = SEED

    dataset = generate_dataset(p, NUM_TRAJECTORIES)
    print("\nDataset size:", len(dataset))

    if len(dataset) == 0:
        print("No valid trajectories generated.")
        return

    train_set, val_set, test_set, train_ids, val_ids, test_ids = split_dataset_by_trajectory(
        dataset,
        train_fraction=TRAIN_FRACTION,
        val_fraction=VAL_FRACTION,
    )

    print("\nTrain trajectory IDs:", train_ids)
    print("Validation trajectory IDs:", val_ids)
    print("Test trajectory IDs:", test_ids)

    print_trajectory_split_summary(train_set, name="Train Set")
    print_trajectory_split_summary(val_set, name="Validation Set")
    print_trajectory_split_summary(test_set, name="Test Set")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_trajectory_dataset(train_set, OUTPUT_DIR / "train")
    save_trajectory_dataset(val_set, OUTPUT_DIR / "val")
    save_trajectory_dataset(test_set, OUTPUT_DIR / "test")

    print(f"\nSaved trajectory dataset to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
