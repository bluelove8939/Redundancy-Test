import os
import math
import torchvision

from redundant_op import generate_ifm, generate_lowered_ifm
from utils.model_presets import imagenet_clust_pretrained


def analyze_with_real_kernel(lowered_if_map, lowered_kernel, W, H, FW, FH, S, P, OW, OH, offset: int=0):
    # Exception Test
    result = {
        'matched': 0,
        'stride exception': 0,
        'out of index exception': 0,
        'step range exception': 0,
        'unknown exception': 0,
    }

    for i1 in range(0, FW * FH, 1):
        for i2 in range(i1 + 1, FW * FH, 1):
            if lowered_kernel[i1] != lowered_kernel[i2]:
                continue

            dv = math.floor(i2 / FW) - math.floor(i1 / FW)
            dh = (i2 % FW) - (i1 % FW)
            dr = (OW * dv + dh) // S

            for lidx, line in enumerate(lowered_if_map):
                # stride exception
                if dv % S != 0 or dh % S != 0:
                    result['stride exception'] += 1
                    continue

                # out of index exception
                oh = (lidx + offset) % OW
                if oh - (dh // S) < 0 or oh - (dh // S) >= OW or (lidx + offset) - dr < 0:
                    result['out of index exception'] += 1
                    continue

                # step range exception
                if lidx - dr < 0:
                    # print(lidx, dr)
                    result['step range exception'] += 1
                    continue

                # unknown exception
                if line[i1] != lowered_if_map[lidx - dr][i2]:
                    result['unknown exception'] += 1
                    continue

                result['matched'] += 1

    return result


def analyze_model_redundancy(config, step_range: int=128, max_iter: int or None=None, save_path: str or None=None):
    save_logs = []
    model_result = {}

    model = config.model_type(quantize=True)
    weights = config.weights

    print(model)

    W, H = 226, 226  # size of input image

    save_logs.append("Test Configs")
    save_logs.append(f"- model: {config.model_type.__name__}")
    save_logs.append(f"- step range: {step_range}")
    save_logs.append(f"- max iter: {max_iter}\n\n")

    for param_name, param in weights.items():
        if 'weight' not in param_name:
            continue

        if 'downsample' in param_name:  # resnet18
            continue

        layer_id = param_name.split('.')[:-1]
        layer_name = '.'.join(layer_id)

        layer = model
        for attr in layer_id:
            layer = getattr(layer, attr)

        C = layer.in_channels
        OC = layer.out_channels
        FW = layer.kernel_size[0]
        FH = layer.kernel_size[1]
        S = layer.stride[0]
        P = layer.padding[0]
        OW = math.floor((W - FW + (2 * P)) / S) + 1
        OH = math.floor((H - FH + (2 * P)) / S) + 1

        save_logs.append(f"{layer_name:30s}  "
                         f"type: {type(layer).__name__:15s}  "
                         f"C: {C:3d}  OC: {OC:3d}  (W, H): {W, H}  (FW, FH): {FW, FH}  S: {S}  P: {P}  (OW, OH): {OW, OH}")

        weights = param.int_repr().detach().numpy()
        lowered_weights = weights.reshape((OC, C, FW*FH))

        if_map = generate_ifm(W, H, P)
        lowered_if_map = generate_lowered_ifm(if_map, W, H, FW, FH, S, P)

        result = {
            'matched': 0,
            'stride exception': 0,
            'out of index exception': 0,
            'step range exception': 0,
            'unknown exception': 0,
        }

        iter_cnt = 0

        for oc_idx in range(OC):
            for c_idx in range(C):
                if max_iter is not None and iter_cnt > max_iter:
                    break

                lowered_kernel = lowered_weights[oc_idx, c_idx]

                for l_offset in range(0, lowered_if_map.shape[0], step_range):
                    lowered_if_map_part = lowered_if_map[l_offset:l_offset+step_range]
                    tmp_result = analyze_with_real_kernel(lowered_if_map_part, lowered_kernel,
                                                          W, H, FW, FH, S, P, OW, OH, offset=l_offset)

                    for key in tmp_result:
                        result[key] += tmp_result[key]

                iter_cnt += 1

        model_result[layer_name] = result

        W, H = OW, OH

    if save_path is not None:
        with open(save_path, 'wt') as file:
            file.write('\n'.join(save_logs))
            file.write('\n\n\n')

            for lname, lresult in model_result.items():
                file.write(f"layer: {lname:30s} |     {'    '.join([f'{k}: {v:8d}' for k, v in lresult.items()])}\n")

    return model_result


if __name__ == '__main__':
    model_name = 'ResNet18'
    # model_name = 'GoogLeNet'
    step_range = 5000

    save_dirname = os.path.join(os.curdir, 'results', 'real_weight_redundancy')
    save_path = os.path.join(save_dirname, f'{model_name}_{step_range}.txt')

    os.makedirs(save_dirname, exist_ok=True)

    result = analyze_model_redundancy(config=imagenet_clust_pretrained[model_name], max_iter=1,
                                      step_range=step_range, save_path=save_path)

    for lname, lresult in result.items():
        print(f"layer: {lname:30s} |     {'    '.join([f'{k}: {v:9d}' for k, v in lresult.items()])}")
