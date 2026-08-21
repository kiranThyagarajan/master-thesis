import numpy as np


def get_params():
    return {
        # Geometry
        "l": 0.5,
        "r": 0.25,  # center-of-mass distance from pivot

        # Mass / inertia
        "m": 2.0,
        "I": 0.03,

        # Gravity
        "g": 9.81,

        # Optional viscous joint damping
        "b": 0.0,

        # Simulation
        "dt": 0.005,
        "T": 5.0,

        # PD gains
        "Kp": 40.0,
        "Kd": 10.0,

        # Horizontal right is zero.
        # Vertical upward is pi/2.
        "q_target": np.deg2rad(55.0),
        "qdot_target": 0.0,
    }