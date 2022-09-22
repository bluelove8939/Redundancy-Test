import math

exception_keys = ['matched', 'stride exception', 'out of index exception',
                              'step range exception', 'unknown exception',]

def analyze_with_real_kernel(lowered_if_map, lowered_kernel, W, H, FW, FH, S, P, OW, OH, offset: int=0):
    # Exception Test
    result = {ek: 0 for ek in exception_keys}

    for i1 in range(0, FW * FH, 1):
        for i2 in range(i1 + 1, FW * FH, 1):
            if lowered_kernel[i1] != lowered_kernel[i2]:
                continue

            dv = math.floor(i2 / FW) - math.floor(i1 / FW)
            dh = (i2 % FW) - (i1 % FW)
            dr = (OW * dv + dh) // S

            for lidx, line in enumerate(lowered_if_map):
                # # stride exception
                # if dv % S != 0 or dh % S != 0:
                #     result['stride exception'] += 1
                #     continue
                #
                # # out of index exception
                # oh = (lidx + offset) % OW
                # if oh - (dh // S) < 0 or oh - (dh // S) >= OW or (lidx + offset) - dr < 0:
                #     result['out of index exception'] += 1
                #     continue

                # step range exception
                if lidx - dr < 0:
                    result['step range exception'] += 1
                    continue

                # unknown exception
                if line[i1] == lowered_if_map[lidx - dr][i2]:
                    result['matched'] += 1
                    continue

                result['unknown exception'] += 1

            break  # check only one redundancy element (duplicated redunadncy)

    return result