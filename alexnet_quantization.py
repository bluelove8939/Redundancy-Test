import os
import torch

from models.model_presets import imagenet_pretrained                             # normal model
from models.tools.quanitzation import QuantizationModule                         # quantization module
from models.tools.imagenet_utils.dataset_loader import val_loader, train_loader  # datasets (imagenet)
from models.tools.imagenet_utils.training import validate                        # validation method (measuring acc)
from models.tools.imagenet_utils.args_generator import args


if __name__ == '__main__':
    # Generate model without quantization
    config = imagenet_pretrained['AlexNet']
    model = config.generate()

    # Quantization setup
    tuning_dataloader = train_loader
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

    # Quantize model
    qmod = QuantizationModule(tuning_dataloader=tuning_dataloader, criterion=criterion, optimizer=optimizer)
    qmodel = qmod.quantize(model=model, citer=10, verbose=1)  # calibration
    # qmodel = qmod.quantize(model=model, citer=0)              # no calibration

    # Save quantized parameters
    dirname = os.path.join(os.curdir, 'model_output')
    filename = 'AlexNet_quantized_tuned_citer_10.pth'

    os.makedirs(dirname, exist_ok=True)
    torch.save(qmodel.state_dict(), os.path.join(dirname, filename))

    # Check acuracy degradation
    validate(val_loader=val_loader, model=qmodel, criterion=criterion, args=args, device='cpu', at_prune=False, pbar_header='')

    # for nm, mod in qmodel.named_modules():
    #     if 'conv' in type(mod).__name__.lower():
    #         print(mod.weight().int_repr().detach())