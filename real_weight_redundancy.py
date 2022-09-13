import math
import torchvision

from redundant_op import generate_ifm, generate_lowered_ifm
from utils.model_presets import imagenet_clust_pretrained


def analyze_model_redundancy(config):
    model = config.model_type(quantize=True)
    weights = config.weights

    W, H = 226, 226  # size of input image

    for param_name, param in weights.items():
        if 'weight' not in param_name:
            continue

        if 'downsample' in param_name:
            continue

        weight = param.int_repr()
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

        print(f"{layer_name:25s}  "
              f"type: {type(layer).__name__:15s}  "
              f"C: {C:3d}  OC: {OC:3d}  (W, H): {W, H}  (FW, FH): {FW, FH}  S: {S}  P: {P}  (OW, OH): {OW, OH}")

        if_map = generate_ifm(W, H, P)
        lowered_ifm = generate_lowered_ifm(if_map, W, H, FW, FH, S, P)

        print(lowered_ifm.shape)

        W, H = OW, OH


if __name__ == '__main__':
    analyze_model_redundancy(config=imagenet_clust_pretrained['ResNet18'])
