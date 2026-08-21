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
from markers import (
    get_marker_observation,
    get_marker_positions,
)
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


def angular_error(q, q_target):
    """
    Return the signed shortest angular difference q - q_target.
    """
    return (q - q_target + np.pi) % (2.0 * np.pi) - np.pi


def state_derivative(t, x, p):
    del t  # autonomous system

    q = x[0]
    qdot = x[1]

    tau = pd_control(q, qdot, p)
    acceleration = qddot(q, qdot, tau, p)

    return np.array([
        qdot,
        acceleration,
    ])


def simulate_trajectory(x0, p):
    dt = p["dt"]
    T = p["T"]

    # This avoids possible floating-point overshoot past T.
    number_of_steps = int(np.round(T / dt))
    time = np.linspace(
        0.0,
        T,
        number_of_steps + 1,
    )

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

    # One actuator
    U = np.zeros((N, 1), dtype=float)

    # Three markers, each with x and y
    Y = np.zeros((N, 3, 2), dtype=float)

    # 6 position values + 6 velocity values
    O = np.zeros((N, 12), dtype=float)

    I = np.zeros(
        (N, 64, 128),
        dtype=np.uint8,
    )

    for k in range(N):
        q = X[k, 0]
        qdot = X[k, 1]

        U[k, 0] = pd_control(q, qdot, p)
        Y[k] = get_marker_positions(q, p)
        O[k] = get_marker_observation(q, qdot, p)
        I[k] = render_markers(Y[k], p)

    return {
        "x0": np.asarray(x0, dtype=float),
        "time": sol.t,
        "X": X,
        "U": U,
        "Y": Y,
        "O": O,
        "I": I,
    }


def find_convergence_index(
    traj,
    p,
    q_tol,
    qdot_tol,
    streak_length,
):
    q = traj["X"][:, 0]
    qdot = traj["X"][:, 1]

    q_error = np.abs(
        angular_error(q, p["q_target"])
    )

    qdot_error = np.abs(
        qdot - p["qdot_target"]
    )

    converged = (
        (q_error <= q_tol)
        & (qdot_error <= qdot_tol)
    )

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

    max_attempts = max(
        num_trajectories * MAX_ATTEMPTS_FACTOR,
        num_trajectories,
    )

    while (
        len(dataset) < num_trajectories
        and attempts < max_attempts
    ):
        attempts += 1

        x0 = sample_initial_condition(p, rng)
        traj = simulate_trajectory(x0, p)

        if traj is None:
            rejected += 1
            print(
                f"Attempt {attempts}: trajectory simulation failed"
            )
            continue

        if len(traj["time"]) < SEQUENCE_LENGTH:
            rejected += 1
            print(
                f"Attempt {attempts}: trajectory too short for "
                f"{SEQUENCE_LENGTH} steps"
            )
            continue

        convergence_idx = find_convergence_index(
            traj,
            p,
            q_tol=CONVERGENCE_Q_TOL,
            qdot_tol=CONVERGENCE_QDOT_TOL,
            streak_length=CONVERGENCE_STREAK,
        )

        if (
            convergence_idx is not None
            and convergence_idx < SEQUENCE_LENGTH
        ):
            rejected += 1
            print(
                f"Attempt {attempts}: rejected, converged too "
                f"early at step {convergence_idx}"
            )
            continue

        dataset.append(
            truncate_trajectory(
                traj,
                SEQUENCE_LENGTH,
            )
        )

        print(
            f"Accepted trajectory {len(dataset)}/"
            f"{num_trajectories} on attempt {attempts}"
        )

    if len(dataset) < num_trajectories:
        raise RuntimeError(
            f"Only collected {len(dataset)} valid trajectories "
            f"after {attempts} attempts."
        )

    print(f"\nRejected trajectories: {rejected}")

    return dataset


def main():
    p = get_params()

    if SEED is not None:
        p["seed"] = SEED

    dataset = generate_dataset(
        p,
        NUM_TRAJECTORIES,
    )

    print("\nDataset size:", len(dataset))

    if not dataset:
        print("No valid trajectories generated.")
        return

    (
        train_set,
        val_set,
        test_set,
        train_ids,
        val_ids,
        test_ids,
    ) = split_dataset_by_trajectory(
        dataset,
        train_fraction=TRAIN_FRACTION,
        val_fraction=VAL_FRACTION,
    )

    print("\nTrain trajectory IDs:", train_ids)
    print("Validation trajectory IDs:", val_ids)
    print("Test trajectory IDs:", test_ids)

    print_trajectory_split_summary(
        train_set,
        name="Train Set",
    )

    print_trajectory_split_summary(
        val_set,
        name="Validation Set",
    )

    print_trajectory_split_summary(
        test_set,
        name="Test Set",
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_trajectory_dataset(
        train_set,
        OUTPUT_DIR / "train",
    )

    save_trajectory_dataset(
        val_set,
        OUTPUT_DIR / "val",
    )

    save_trajectory_dataset(
        test_set,
        OUTPUT_DIR / "test",
    )

    print(
        f"\nSaved trajectory dataset to: "
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()