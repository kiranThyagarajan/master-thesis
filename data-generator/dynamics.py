import numpy as np


def M_matrix(q, p):
    _, q2 = q

    m1, m2 = p["m1"], p["m2"]
    l1 = p["l1"]
    r1, r2 = p["r1"], p["r2"]
    I1, I2 = p["I1"], p["I2"]

    a = I1 + I2 + m1 * r1**2 + m2 * (l1**2 + r2**2)
    b = m2 * l1 * r2
    d = I2 + m2 * r2**2

    return np.array([
        [a + 2.0 * b * np.cos(q2), d + b * np.cos(q2)],
        [d + b * np.cos(q2),       d]
    ])


def C_times_qdot(q, qdot, p):
    _, q2 = q
    q1d, q2d = qdot

    b = p["m2"] * p["l1"] * p["r2"]
    h = b * np.sin(q2)

    return np.array([
        -h * (2.0 * q1d * q2d + q2d**2),
         h * q1d**2
    ])


def G_vector(q, p):
    q1, q2 = q

    g = p["g"]
    m1, m2 = p["m1"], p["m2"]
    l1, r1, r2 = p["l1"], p["r1"], p["r2"]

    return np.array([
        (m1 * r1 + m2 * l1) * g * np.cos(q1) + m2 * g * r2 * np.cos(q1 + q2),
        m2 * g * r2 * np.cos(q1 + q2)
    ])


def qddot(q, qdot, tau, p):
    M = M_matrix(q, p)
    Cqdot = C_times_qdot(q, qdot, p)
    G = G_vector(q, p)

    return np.linalg.solve(M, tau - Cqdot - G)