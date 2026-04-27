import numpy as np


def get_params():
    return {
        # geometry
        "l1": 0.5,
        "l2": 0.4,
        "r1": 0.25,
        "r2": 0.20,

        # mass / inertia
        "m1": 2.0,
        "m2": 1.5,
        "I1": 0.03,
        "I2": 0.02,

        # gravity
        "g": 9.81,

        # simulation
        "dt": 0.005,
        "T": 5.0,

        # hand-picked PD gains
        "Kp": np.diag([40.0, 30.0]),
        "Kd": np.diag([10.0, 8.0]),

        # fixed target state
        "q_target": np.array([np.deg2rad(120.0), np.deg2rad(-90.0)]),
        "qdot_target": np.array([0.0, 0.0]),
    }
