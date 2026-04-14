from dynamics import G_vector


def pd_control(q, qdot, p):
    q_target = p["q_target"]
    qdot_target = p["qdot_target"]

    e = q_target - q
    edot = qdot_target - qdot

    tau = p["Kp"] @ e + p["Kd"] @ edot + G_vector(q, p)
    return tau