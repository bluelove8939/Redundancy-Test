import numpy as np


def generate_lowered_ifm(if_map: np.ndarray, W, H, FW, FH, S=1, P=0):
    lowered_if_map = []
    for rp in range(0, H - FH + (2*P) + 1, S):
        for cp in range(0, W - FW + (2*P) + 1, S):
            lowered_if_map.append(list(if_map[rp:rp + FH, cp:cp + FW].flatten()))

    lowered_if_map = np.array(lowered_if_map)

    return lowered_if_map