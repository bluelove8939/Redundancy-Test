import numpy as np


def lowering_ifm(ifm, W, H, FW, FH, S):
    lowered_if_map = []
    for rp in range(0, H - FH + 1, S):
        for cp in range(0, W - FW + 1, S):
            lowered_if_map.append(list(ifm[rp:rp + FH, cp:cp + FW].flatten()))

    return np.array(lowered_if_map)

def lowering_weight(weight):
    return weight.flatten()