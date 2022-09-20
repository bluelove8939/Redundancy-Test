import os

import torch
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch.nn.utils.prune as prune

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
    # Test configuration
    model_name = 'AlexNet'
    config = imagenet_pretrained[model_name]
    model = config.generate().to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 0.0001,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Check normal accuracy and loss
    normal_acc, normal_loss = validate(val_loader=test_loader, model=model, criterion=criterion, args=args)

    # # Pruning
    # # tuning_size = int(len(train_dataset) * 0.05)
    # # tuning_dataset, _ = torch.utils.data.random_split(train_dataset, lengths=[
    # #     tuning_size, len(train_dataset) - tuning_size])
    # # tuning_loader = torch.utils.data.DataLoader(
    # #     tuning_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    # #     num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    #
    # pmodule = PruneModule(train_dataset=train_dataset, optimizer=optimizer, loss_fn=criterion, tuning_ratio=0.05)
    # pmodel = pmodule.prune_imagenet_model(model, target_amount=0.7, threshold=0.5, step=0.1, max_iter=10,
    #                                       pass_normal=False, verbose=1, args=args)
    # pmodule.remove_prune_model(pmodel)

    def remove_prune_model(self, module: torch.nn.Module):
        for sub_idx, sub_module in module._modules.items():
            if isinstance(sub_module, torch.nn.Conv2d):
                prune.remove(sub_module, 'weight')
            elif isinstance(sub_module, torch.nn.Module):
                self.remove_prune_model(sub_module)

    def prune_layer(self, model, step):
        for sub_idx, sub_module in model._modules.items():
            if isinstance(sub_module, torch.nn.Conv2d):
                prune.l1_unstructured(sub_module, 'weight', amount=step)
            elif isinstance(sub_module, torch.nn.Module):
                self.prune_layer(sub_module, step)

    # Check pruned accuracy and loss
    pruned_acc, pruned_loss = validate(val_loader=test_loader, model=model, criterion=criterion, args=args)

    print(f"normal) Acc: {normal_acc}  Loss: {normal_loss}")
    print(f"pruned) Acc: {pruned_acc}  Loss: {pruned_loss}")

    # Save state dictionary
    save_dirname = os.path.join('C://', 'torch_data', 'pruned_weights')
    save_filename = model_name + '.pth'
    save_path = os.path.join(save_dirname, save_filename)

    torch.save(pmodel.state_dict(), save_path)