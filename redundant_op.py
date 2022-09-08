import os
import math
import tqdm
import numpy as np


def generate_ifm(W, H):
    if_map = np.arange(1, W*H+1, 1).reshape((H, W))
    return if_map

def generate_lowered_ifm(if_map: np.ndarray, W, H, FW, FH, S):
    lowered_if_map = []
    for rp in range(0, H - FH + 1, S):
        for cp in range(0, W - FW + 1, S):
            lowered_if_map.append(list(if_map[rp:rp + FH, cp:cp + FW].flatten()))

    lowered_if_map = np.array(lowered_if_map)

    return lowered_if_map

def test_redundancy(W, H, FW, FH, S):
    OW = math.floor((W - FW) / S) + 1
    OH = math.floor((H - FH) / S) + 1

    if_map = generate_ifm(W, H)
    lowered_if_map = generate_lowered_ifm(if_map, W, H, FW, FH, S)

    # Exception Test
    result = {
        'matched': 0,
        'stride exception': 0,
        'out of index exception': 0,
        'unknown exception': 0,
    }

    for i1 in range(0, FW * FH, 1):
        for i2 in range(i1 + 1, FW * FH, 1):
            dv = math.floor(i2 / FW) - math.floor(i1 / FW)
            dh = (i2 % FW) - (i1 % FW)
            dr = (OW * dv + dh) // S

            for lidx, line in enumerate(lowered_if_map):
                # stride exception
                if dv % S != 0 or dh % S != 0:
                    result['stride exception'] += 1
                    continue

                # out of index exception
                oh = lidx % OW
                if oh - (dh // S) < 0 or oh - (dh // S) >= OW or lidx - dr < 0:
                    result['out of index exception'] += 1
                    continue

                # unknown exception
                if line[i1] != lowered_if_map[lidx - dr][i2]:
                    result['unknown exception'] += 1
                    continue

                result['matched'] += 1

    return result


if __name__ == '__main__':
    W = 64
    H = 64
    FW = 3
    FH = 3
    P = 0
    S = 1
    OW = math.floor((W + 2 * P - FW) / S) + 1
    OH = math.floor((H + 2 * P - FH) / S) + 1

    if_map = generate_ifm(W, H)
    lowered_if_map = generate_lowered_ifm(if_map, W, H, FW, FH)

    # Exception Test
    result = {
        'pattern matched': 0,
        'stride exception': 0,
        'out of index exception': 0,
        'unknown exception': 0,
    }
    logs = {
        'pattern matched': [],
        'stride exception': [],
        'out of index exception': [],
        'unknown exception': [],
    }

    for i1 in range(0, FW * FH, 1):
        for i2 in range(i1 + 1, FW * FH, 1):
            d = i2 - i1
            dv = math.floor(i2 / FW) - math.floor(i1 / FW)
            dh = (i2 % FW) - (i1 % FW)
            dr = (OW * dv + dh) // S

            for lidx, line in enumerate(lowered_if_map):
                # stride exception
                if dv % S != 0 or dh % S != 0:
                    logs['stride exception'].append(
                        f"lidx: {lidx:3d}  (d, dh, dv, dr): {d, dh, dv, dr}  "
                        f"(e[{lidx}][{i1}], e[{lidx - dr}][{i2}]): {line[i1], lowered_if_map[lidx - dr][i2]}"
                    )
                    result['stride exception'] += 1
                    continue

                # out of index exception
                oh = lidx % OW
                if oh - (dh // S) < 0 or oh - (dh // S) >= OW or lidx - dr < 0:
                    logs['out of index exception'].append(
                        f"lidx: {lidx:3d}  oh: {oh:2d}  (d, dh, dv, dr): {d, dh, dv, dr}  "
                        f"e[{lidx}][{i1}]: {line[i1]}"
                    )
                    result['out of index exception'] += 1
                    continue

                # unknown exception
                if line[i1] != lowered_if_map[lidx - dr][i2]:
                    logs['unknown exception'].append(
                        f"lidx: {lidx:3d}  oh: {oh:2d}  (d, dh, dv, dr): {d, dh, dv, dr}  "
                        f"(e[{lidx}][{i1}], e[{lidx - dr}][{i2}]): {line[i1], lowered_if_map[lidx - dr][i2]}"
                    )
                    result['unknown exception'] += 1
                    continue

                logs['pattern matched'].append(
                    f"lidx: {lidx:3d}  oh: {oh:2d}  (d, dh, dv, dr): {d, dh, dv, dr}  "
                    f"(e[{lidx}][{i1}], e[{lidx - dr}][{i2}]): {line[i1], lowered_if_map[lidx - dr][i2]}"
                )
                result['pattern matched'] += 1

    with open(os.path.join(os.curdir, 'redundant_op_ulog.txt'), 'wt') as file:
        file.write('\n\n'.join(
            [f"{exception_name}\n" + ('\n'.join(exception_log) if len(exception_log) > 0 else 'No exception') for
             exception_name, exception_log in logs.items()]))

    with open(os.path.join(os.curdir, 'redundant_op_lifm.txt'), 'wt') as file:
        file.write('\n'.join(
            ['IFM'] +
            [f"lidx: {lidx:3d} -> " + '  '.join(map(lambda x: f"{x:3d}", line)) for lidx, line in enumerate(if_map)]
        ))

        file.write('\n'.join(
            ['\n\nLowered IFM'] +
            [f"lidx: {lidx:3d} -> " + '  '.join(map(lambda x: f"{x:3d}", line)) for lidx, line in
             enumerate(lowered_if_map)]
        ))

    print(result)
