import os
import torch

from models.model_presets import imagenet_pretrained
from models.tools.quanitzation import QuantizationModule
from models.tools.imagenet_utils.dataset_loader import val_loader, train_loader
from models.tools.imagenet_utils.training import validate
from models.tools.imagenet_utils.args_generator import args


# device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == '__main__':
    config = imagenet_pretrained['VGG16']
    model = config.generate()

    tuning_dataloader = train_loader
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

    qmod = QuantizationModule(tuning_dataloader=tuning_dataloader, criterion=criterion, optimizer=optimizer)
    qmodel = qmod.quantize(model=model, citer=10, verbose=1)

    dirname = os.path.join(os.curdir, 'model_output')
    filename = 'VGG16_quantized_tuned_citer_10.pth'

    os.makedirs(dirname, exist_ok=True)

    torch.save(qmodel.state_dict(), os.path.join(dirname, filename))

    validate(val_loader=val_loader, model=qmodel, criterion=criterion, args=args, device='cpu', at_prune=False, pbar_header='')