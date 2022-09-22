import os

import torch
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch.nn.utils.prune as prune

from models.tools.pruning import PruneModule
from models.model_presets import imagenet_pruned
from models.tools.imagenet_utils.args_generator import args
from models.tools.imagenet_utils.training import train, validate


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
    # Test configuration
    model_name = 'AlexNet'
    config = imagenet_pruned[model_name]
    normal_model = config.generate(load_chkpoint=False).to(device)
    pruned_model = config.generate(load_chkpoint=True).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    normal_acc, normal_loss = validate(val_loader=test_loader, model=normal_model, criterion=criterion, args=args, pbar_header='normal acc')
    pruned_acc, pruned_loss = validate(val_loader=test_loader, model=pruned_model, criterion=criterion, args=args, pbar_header='pruned acc')

    # Check pruned accuracy and loss
    print(f"normal) Acc: {normal_acc}  Loss: {normal_loss}")
    print(f"pruned) Acc: {pruned_acc}  Loss: {pruned_loss}")

    # # Save state dictionary
    # save_dirname = os.path.join('C://', 'torch_data', 'pruned_weights')
    # save_filename = model_name + '.pth'
    # save_path = os.path.join(save_dirname, save_filename)
    #
    # torch.save(model.state_dict(), save_path)