import os
import torch
import torchvision

from utils.model_presets import imagenet_clust_pretrained

for name, config in imagenet_clust_pretrained.items():
    print(name, config.generate())