import numpy as np

# fractions along each link
link1_fracs = [0.25, 0.5, 0.75]
link2_fracs = [0.25, 0.5, 0.75]

def get_marker_positions(q, p):
    """
    Compute marker positions for a 2-link planar arm.

    Returns:
        markers: array of shape (m, 2)
                 each row = [x, y]
    """

    q1, q2 = q
    l1, l2 = p["l1"], p["l2"]

    markers_positions = []

    # --- Link 1 markers ---
    for frac in link1_fracs:
        d = frac * l1

        x = d * np.cos(q1)
        y = d * np.sin(q1)

        markers_positions.append([x, y])

    # elbow position (start of link 2)
    x_elbow = l1 * np.cos(q1)
    y_elbow = l1 * np.sin(q1)

    # --- Link 2 markers ---
    for frac in link2_fracs:
        s = frac * l2

        x = x_elbow + s * np.cos(q1 + q2)
        y = y_elbow + s * np.sin(q1 + q2)

        markers_positions.append([x, y])

    return np.array(markers_positions) # shape (6, 2) -> 6 markers, each with (x, y)

def get_marker_velocities(q, qdot, p):
    """
    Compute marker velocities for a 2-link planar arm.

    Returns:
        marker_velocities: array of shape (m, 2)
                           each row = [xdot, ydot]
    """

    q1, q2 = q
    q1dot, q2dot = qdot
    l1, l2 = p["l1"], p["l2"]

    marker_velocities = []

    # --- Link 1 marker velocities ---
    for frac in link1_fracs:
        d = frac * l1

        xdot = -d * np.sin(q1) * q1dot
        ydot =  d * np.cos(q1) * q1dot

        marker_velocities.append([xdot, ydot])

    # --- Link 2 marker velocities ---
    for frac in link2_fracs:
        s = frac * l2

        xdot = -l1 * q1dot * np.sin(q1) - s * np.sin(q1 + q2) * (q1dot + q2dot)
        ydot =  l1 * q1dot * np.cos(q1) + s * np.cos(q1 + q2) * (q1dot + q2dot)

        marker_velocities.append([xdot, ydot])

    return np.array(marker_velocities)  # shape (6, 2) -> 6 markers, each with (xdot, ydot)


def get_marker_observation(q, qdot, p):
    marker_positions = get_marker_positions(q, p)
    marker_positions = marker_positions.flatten()  # shape (12,) -> 6 markers, each with (x, y)
    marker_velocities = get_marker_velocities(q, qdot, p)
    marker_velocities = marker_velocities.flatten()  # shape (12,) -> 6 markers, each with (xdot, ydot)
    return np.concatenate([marker_positions, marker_velocities], axis=0)
