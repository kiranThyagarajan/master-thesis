import numpy as np


# Fractions along the pendulum link
link_fracs = [0.25, 0.5, 0.75]


def get_marker_positions(q, p):
    """
    Compute marker positions for a single planar pendulum.

    Returns:
        markers: array with shape (3, 2)
    """
    l = p["l"]

    marker_positions = []

    for frac in link_fracs:
        distance = frac * l

        x = distance * np.cos(q)
        y = distance * np.sin(q)

        marker_positions.append([x, y])

    return np.asarray(marker_positions, dtype=float)


def get_marker_velocities(q, qdot, p):
    """
    Compute marker velocities.

    For a marker at distance d:

        x = d*cos(q)
        y = d*sin(q)

        xdot = -d*sin(q)*qdot
        ydot =  d*cos(q)*qdot

    Returns:
        velocities: array with shape (3, 2)
    """
    l = p["l"]

    marker_velocities = []

    for frac in link_fracs:
        distance = frac * l

        xdot = -distance * np.sin(q) * qdot
        ydot = distance * np.cos(q) * qdot

        marker_velocities.append([xdot, ydot])

    return np.asarray(marker_velocities, dtype=float)


def get_marker_observation(q, qdot, p):
    """
    Return a 12-dimensional observation:

        [marker positions, marker velocities]
    """
    positions = get_marker_positions(q, p).flatten()
    velocities = get_marker_velocities(q, qdot, p).flatten()

    return np.concatenate([positions, velocities], axis=0)


def physical_state_from_marker_observation(obs, p):
    """
    Reconstruct [q, qdot] from a 12-dimensional marker observation.

    Observation layout:
        first 6 values: 3 marker positions
        last 6 values: 3 marker velocities
    """
    obs = np.asarray(obs, dtype=float)

    if obs.shape != (12,):
        raise ValueError(
            f"Expected observation with shape (12,), received {obs.shape}"
        )

    positions = obs[:6].reshape(3, 2)
    velocities = obs[6:].reshape(3, 2)

    frac = link_fracs[0]
    distance = frac * p["l"]

    marker_position = positions[0]
    marker_velocity = velocities[0]

    q = np.arctan2(
        marker_position[1],
        marker_position[0],
    )

    # Tangential velocity projected onto [-sin(q), cos(q)]
    qdot = (
        -np.sin(q) * marker_velocity[0]
        + np.cos(q) * marker_velocity[1]
    ) / distance

    return np.array([q, qdot], dtype=float)