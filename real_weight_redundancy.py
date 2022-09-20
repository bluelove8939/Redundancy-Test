import os
import math
import tqdm
import torch
import numpy as np

from redundant_op import generate_ifm, generate_lowered_ifm
from models.model_presets import imagenet_clust_pretrained, ClustModelConfig


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


def analyze_model_redundancy(config: ClustModelConfig, step_range: int=128,
                             max_iter: int or None=None, save_path: str or None=None):
    # Test header
    save_logs = list()
    save_logs.append("Test Configs")
    save_logs.append(f"- test: real weight redundancy test")
    save_logs.append(f"- model: {config.model_type.__name__}")
    save_logs.append(f"- step range: {step_range}")
    save_logs.append(f"- max iter: {max_iter}\n\n")

    print("\n\nTest Configs")
    print(f"- test: real weight redundancy test")
    print(f"- model: {config.model_type.__name__}")
    print(f"- step range: {step_range}")
    print(f"- max iter: {max_iter}\n")

    model_result = {}
    model = config.model_type(quantize=True, weights=config.default_weights)
    weights = config.weights

    # Extract input tensor shape of each layer
    W, H = 224, 224  # size of input image
    dummy_image = torch.tensor(np.zeros(shape=(1, 3, H, W), dtype=np.dtype('float32')))

    input_shape_dict = {}
    total_operation_dict = {}

    def generate_input_shape_hook(input_shape_dict, layer_name):
        def hook(model, input_tensor, output_tensor):
            input_shape_dict[layer_name] = input_tensor[0].int_repr().shape
        return hook

    for lname, layer in model.named_modules():
        if 'conv' in type(layer).__name__.lower():
            layer.register_forward_hook(generate_input_shape_hook(input_shape_dict, lname))

    model.eval()
    model(dummy_image)

    # Redundancy test using real weight tensor
    for param_name, param in weights.items():
        if 'weight' not in param_name:  # iff the given parameter is weight
            continue

        # Extract details of each convolution layer
        layer_id = param_name.split('.')[:-1]
        layer_name = '.'.join(layer_id)

        layer = model
        for attr in layer_id:
            layer = getattr(layer, attr)

        if 'conv' not in type(layer).__name__.lower():  # iff the given layer is convolution layer
            continue

        _, C, W, H = input_shape_dict[layer_name]
        OC = layer.out_channels
        FW, FH = layer.kernel_size
        S = layer.stride[0]
        P = layer.padding[0]
        OW = math.floor((W - FW + (2 * P)) / S) + 1
        OH = math.floor((H - FH + (2 * P)) / S) + 1

        # if FW == 1 and FH == 1:  # if kernel shape is (1, 1), redundancy cannot occur
        #     continue

        save_logs.append(f"{layer_name:30s}  "
                         f"type: {type(layer).__name__:15s}  "
                         f"C: {C:3d}  OC: {OC:3d}  (W, H): {W, H}  (FW, FH): {FW, FH}  "
                         f"S: {S}  P: {P}  (OW, OH): {OW, OH}")

        # Extract weight and generate lowered weight
        weights = param.int_repr().detach().numpy()
        lowered_weights = weights.reshape((OC, C, FW*FH))

        # Generate lowered input feature map
        if_map = generate_ifm(W, H, P)
        lowered_if_map = generate_lowered_ifm(if_map, W, H, FW, FH, S, P)
        total_operation_dict[layer_name] = 0

        # Start testing
        result = {ek: 0 for ek in exception_keys}
        iter_cnt = 0
        step_range = lowered_if_map.shape[0] if step_range is None else step_range

        with tqdm.tqdm(ncols=100, total=(min(max_iter, OC*C) if max_iter is not None else OC*C),
                       desc=f"{layer_name:30s}", leave=False) as pbar:
            for oc_idx in range(OC):
                for c_idx in range(C):  # shape of lowered weights: (OC, C, FH*FW)
                    if max_iter is not None and iter_cnt > max_iter:
                        break

                    lowered_kernel = lowered_weights[oc_idx, c_idx]

                    for l_offset in range(0, lowered_if_map.shape[0], step_range):  # split IFM with step range
                        lowered_if_map_part = lowered_if_map[l_offset:min(l_offset+step_range, lowered_if_map.shape[0])]
                        tmp_result = analyze_with_real_kernel(lowered_if_map_part, lowered_kernel,
                                                              W, H, FW, FH, S, P, OW, OH, offset=l_offset)
                        total_operation_dict[layer_name] += lowered_if_map_part.shape[0] * lowered_if_map_part.shape[1]

                        for key in tmp_result:
                            result[key] += tmp_result[key]

                    iter_cnt += 1
                    pbar.update(1)

        model_result[layer_name] = result

    # Save test result as a text file (if needed)
    if save_path is not None:
        with open(save_path, 'wt') as file:
            # Logs are saved as comments
            file.write('# ' + '\n# '.join(save_logs))
            file.write('\n\n\n')

            # Test results are saved as CSV format
            file.write(f"{'layer name':30s}, {'total':20s}, {', '.join([f'{ek:30s}' for ek in exception_keys])}\n")
            for lname, lresult in model_result.items():
                file.write(f"{lname:30s}, {total_operation_dict[lname]:20d}, {', '.join([f'{lresult[ek]:30d}' for ek in exception_keys])}\n")

    # Print result
    for lname, lresult in model_result.items():
        print(f"{lname:30s}  {'    '.join([f'{k}: {v:9d}' for k, v in lresult.items()])}")

    return model_result


if __name__ == '__main__':
    max_iter = None

    save_dirname = os.path.join(os.curdir, 'results', 'real_weight_redundancy')
    os.makedirs(save_dirname, exist_ok=True)

    for model_name in imagenet_clust_pretrained.keys():
        # if 'resnet' not in model_name.lower():
        #     continue

        for step_range in [32, 64, 128, 256, None]:
            save_path = os.path.join(save_dirname, f'{model_name}_{step_range}.csv')
            result = analyze_model_redundancy(config=imagenet_clust_pretrained[model_name], max_iter=max_iter,
                                              step_range=step_range, save_path=save_path)
