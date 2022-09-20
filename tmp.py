import os

import torch
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models.tools.pruning import PruneModule
from models.model_presets import imagenet_pretrained
from models.tools.imagenet_utils.args_generator import args
from models.tools.imagenet_utils.training import validate


device = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset configuration
dataset_dirname = args.data
if not os.path.isdir(dataset_dirname):
    dataset_dirname = os.path.join('C://', 'torch_data', 'imagenet')

train_dataset = datasets.ImageFolder(
        os.path.join(dataset_dirname, 'train'),
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))

test_dataset = datasets.ImageFolder(
        os.path.join(dataset_dirname, 'val'),
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))

if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False, drop_last=True)
else:
    train_sampler = None
    val_sampler = None

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    num_workers=args.workers, pin_memory=True, sampler=train_sampler)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True, sampler=val_sampler)


if __name__ == '__main__':
    model_name = 'AlexNet'
    config = imagenet_pretrained[model_name]
    model = config.generate().to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    tuning_dataset, _ = torch.utils.data.random_split(train_dataset, lengths=[
        int(len(train_dataset) * 0.1), len(train_dataset) - int(len(train_dataset) * 0.1)])
    tuning_loader = torch.utils.data.DataLoader(
        tuning_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    pmodule = PruneModule(tuning_dataloader=tuning_loader, optimizer=optimizer, loss_fn=criterion)
    pmodel = pmodule.prune_model(model, target_amount=0.7, threshold=1, step=0.2, max_iter=5, pass_normal=False, verbose=1)

    print(pmodel.state_dict)

    validate(val_loader=test_loader, model=model,  criterion=criterion, args=args)
    validate(val_loader=test_loader, model=pmodel, criterion=criterion, args=args)

    save_dirname = os.path.join('C://', 'torch_data', 'pruned_weights')
    save_filename = model_name + '.pth'
    save_path = os.path.join(save_dirname, save_filename)

    torch.save(pmodel.state_dict(), save_path)