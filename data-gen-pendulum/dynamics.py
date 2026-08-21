import numpy as np


def M_scalar(p):
    """
    Effective inertia of the pendulum about the pivot.
    """
    return p["I"] + p["m"] * p["r"]**2


def G_scalar(q, p):
    """
    Gravity term for an angle measured from the horizontal.

    q = 0       -> horizontal right
    q = pi / 2  -> vertical upward
    """
    return p["m"] * p["g"] * p["r"] * np.cos(q)


def damping_torque(qdot, p):
    """
    Viscous joint damping term.
    """
    return p.get("b", 0.0) * qdot


def qddot(q, qdot, tau, p):
    """
    Single-pendulum equation:

        M * qddot + b*qdot + G(q) = tau
    """
    M = M_scalar(p)
    G = G_scalar(q, p)
    damping = damping_torque(qdot, p)

    return (tau - damping - G) / M