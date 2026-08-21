import numpy as np

from dynamics import G_scalar


def wrap_angle(angle):
    """
    Wrap an angle to [-pi, pi).

    This makes the controller follow the shortest angular direction.
    """
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def pd_control(q, qdot, p):
    q_target = p["q_target"]
    qdot_target = p["qdot_target"]

    # Shortest angular error
    error = wrap_angle(q_target - q)
    velocity_error = qdot_target - qdot

    # Gravity-compensated PD control
    tau = (
        p["Kp"] * error
        + p["Kd"] * velocity_error
        + G_scalar(q, p)
    )

    return float(tau)