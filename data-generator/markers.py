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


def physical_state_from_marker_observation(obs, p):
    """
    Reconstruct [q1, q2, q1dot, q2dot] from a 24D marker observation.

    The observation layout is:
    - first 12 values: 6 marker positions flattened as [x, y]
    - last 12 values: 6 marker velocities flattened as [xdot, ydot]
    """
    positions = obs[:12].reshape(6, 2)
    velocities = obs[12:].reshape(6, 2)

    l1 = p["l1"]
    l2 = p["l2"]
    link1_frac = link1_fracs[0]
    link2_frac = link2_fracs[0]

    d1 = link1_frac * l1
    s2 = link2_frac * l2

    p_link1 = positions[0]
    v_link1 = velocities[0]

    q1 = np.arctan2(p_link1[1], p_link1[0])

    elbow = p_link1 / link1_frac
    p_link2_rel = positions[len(link1_fracs)] - elbow

    q12 = np.arctan2(p_link2_rel[1], p_link2_rel[0])
    q2 = q12 - q1

    q1dot = (-np.sin(q1) * v_link1[0] + np.cos(q1) * v_link1[1]) / d1

    v_elbow = np.array([-l1 * np.sin(q1) * q1dot, l1 * np.cos(q1) * q1dot])

    v_link2_rel = velocities[len(link1_fracs)] - v_elbow
    q12dot = (-np.sin(q12) * v_link2_rel[0] + np.cos(q12) * v_link2_rel[1]) / s2

    q2dot = q12dot - q1dot

    return np.array([q1, q2, q1dot, q2dot])
