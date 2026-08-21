import numpy as np


def forward_kinematics(q, p):
    """
    Return the pendulum endpoint position.

    q = 0       -> endpoint at (l, 0)
    q = pi / 2  -> endpoint at (0, l)
    """
    l = p["l"]

    endpoint = np.array([
        l * np.cos(q),
        l * np.sin(q),
    ])

    return endpoint


def workspace_ok(q, p, tol=1e-9):
    """
    Check whether the full pendulum remains above the ground.

    Because the link is straight and begins at y=0, requiring its endpoint
    to have y >= 0 is sufficient.
    """
    endpoint = forward_kinematics(q, p)
    return endpoint[1] >= -tol


def angular_distance(q1, q2):
    """
    Smallest absolute angular distance between two angles.
    """
    error = (q1 - q2 + np.pi) % (2.0 * np.pi) - np.pi
    return abs(error)


def far_from_target(q, q_target, min_dist_deg=20.0):
    min_dist = np.deg2rad(min_dist_deg)
    return angular_distance(q, q_target) > min_dist


def sample_initial_condition(p, rng=None):
    """
    Sample [q, qdot].

    Since the link must initially be above the ground, q is sampled from
    approximately 0 to pi.
    """
    if rng is None:
        rng = np.random.default_rng()

    for _ in range(1000):
        q = rng.uniform(
            np.deg2rad(10.0),
            np.deg2rad(170.0),
        )

        if not workspace_ok(q, p):
            continue

        if not far_from_target(q, p["q_target"]):
            continue

        qdot = 0.0

        return np.array([q, qdot], dtype=float)

    raise RuntimeError("Could not sample a valid initial condition")