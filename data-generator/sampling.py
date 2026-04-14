import numpy as np


def forward_kinematics(q, p): # finding the position of the elbow and ee
    q1, q2 = q
    l1, l2 = p["l1"], p["l2"]

    elbow = np.array([
        l1 * np.cos(q1),
        l1 * np.sin(q1)
    ])

    ee = np.array([
        l1 * np.cos(q1) + l2 * np.cos(q1 + q2),
        l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
    ])

    return elbow, ee # ee -> end-effector


def workspace_ok(q, p, tol=1e-9): # check if the elbow and ee are above the ground (y >= 0)
    elbow, ee = forward_kinematics(q, p)
    return elbow[1] >= -tol and ee[1] >= -tol


def not_near_singularity(q, eps=0.2): # check if q2 != 0 or pi
    _, q2 = q
    return abs(np.sin(q2)) > eps


def far_from_target(q, q_target, min_dist_deg=10.0): # q - q_target > min_dist
    min_dist = np.deg2rad(min_dist_deg)
    return np.linalg.norm(q - q_target) > min_dist


def sample_initial_condition(p, rng=None): #initialize following the constraints above
    if rng is None:
        rng = np.random.default_rng()

    for _ in range(1000):
        q1 = rng.uniform(np.deg2rad(20), np.deg2rad(160))
        q2 = rng.uniform(np.deg2rad(-120), np.deg2rad(120))

        q = np.array([q1, q2])

        if not not_near_singularity(q):
            continue

        if not workspace_ok(q, p):
            continue

        if not far_from_target(q, p["q_target"]):
            continue

        qdot = np.array([0.0, 0.0])  # safe start

        return np.hstack((q, qdot)) # [q1, q2, qdot1, qdot2]

    raise RuntimeError("Could not sample valid initial condition")
from params import get_params
par = get_params()
for i in range(10):

    print(sample_initial_condition(p=par))


