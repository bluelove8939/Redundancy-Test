import os
import torch

from models.model_presets import imagenet_pretrained
from models.tools.quanitzation import QuantizationModule
from models.tools.imagenet_utils.dataset_loader import val_loader


# device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == '__main__':
    config = imagenet_pretrained['VGG16']
    model = config.generate()

    qmod = QuantizationModule()
    qmodel = qmod.quantize(model=model)

    for name, layer in qmodel.named_modules():
        if 'conv' in type(layer).__name__.lower():
            layer.register_forward_hook(lambda x, y, z: print(name + " called!"))

    iter_cnt = 0
    max_iter = 1

    if __name__ == '__main__':
        for X, y in val_loader:
            if iter_cnt >= max_iter:
                break
            else:
                iter_cnt += 1

            X, y = X.to('cpu'), y.to('cpu')
            qmodel(X)

    torch.save(qmodel.state_dict(), os.path.join(os.curdir, 'model_output', 'VGG16_quantized.pth'))