import os
import abc
import torch
import torchvision


class _MetaModelConfig(metaclass=abc.ABCMeta):
    def __init__(self, model_type, weights=None, default_traces=tuple()):
        self.model_type = model_type
        self.weights = weights
        self.traces = list(default_traces)

    @abc.abstractmethod
    def generate(self):
        pass

    def traced_layers(self, model):
        layers = {}

        for tr in self.traces:
            target = model
            for attr in tr.split('.'):
                target = getattr(target, attr)
            layers[tr.replace('.', '_')] = target

        return layers

class ModelConfig(_MetaModelConfig):
    def __init__(self, model_type, weights=None, default_traces=tuple()):
        super(ModelConfig, self).__init__(model_type, weights, default_traces)

    def generate(self):
        return self.model_type(weights=self.weights)

class QuantModelConfig(_MetaModelConfig):
    def __init__(self, model_type, weights=None, default_traces=tuple()):
        super(QuantModelConfig, self).__init__(model_type, weights, default_traces)

    def generate(self):
        return self.model_type(weights=self.weights, quantize=True)

class ClustModelConfig(_MetaModelConfig):
    def __init__(self, model_type, weights=None, default_traces=tuple(), default_weights=None):
        super(ClustModelConfig, self).__init__(model_type, weights, default_traces)
        self.default_weights = default_weights

    def generate(self):
        model = self.model_type(quantize=True, weights=self.default_weights)
        model.load_state_dict(self.weights)
        return model

class ChkpointModelConfig(_MetaModelConfig):
    def __init__(self, model_type, chkpoint, weights=None, default_traces=tuple(), default_weights=None):
        super(ChkpointModelConfig, self).__init__(model_type, weights, default_traces)
        self.chkpoint = chkpoint
        self.default_weights = default_weights

    def generate(self, load_chkpoint=False):
        model = self.model_type(weights=self.weights)
        if load_chkpoint:
            model.load_state_dict(torch.load(self.chkpoint))
        return model

    def get_chkpoint(self):
        return torch.load(self.chkpoint)


imagenet_pretrained = {
    'ResNet50': ModelConfig(
        torchvision.models.resnet50,
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1,
        default_traces=('conv1', 'layer1', 'layer2', 'layer3', 'layer4',),
    ),
    'ResNet34': ModelConfig(
        torchvision.models.resnet34,
        weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1,
        default_traces=('conv1', 'layer1', 'layer2', 'layer3', 'layer4',),
    ),
    'ResNet18': ModelConfig(
        torchvision.models.resnet18,
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
        default_traces=('conv1', 'layer1', 'layer2', 'layer3', 'layer4',),
    ),
    'AlexNet': ModelConfig(
        torchvision.models.alexnet,
        weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1.IMAGENET1K_V1,
        default_traces=('features.1', 'features.4', 'features.7', 'features.9', 'features.11', 'avgpool',
                        'classifier.2', 'classifier.5',),
    ),
    'VGG16': ModelConfig(
        torchvision.models.vgg16,
        weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1,
        default_traces=('features.1', 'features.3', 'features.6', 'features.8', 'features.11', 'features.13',
                        'features.15', 'features.18', 'features.20', 'features.22', 'features.25', 'features.27',
                        'features.29', 'avgpool', 'classifier.1', 'classifier.4',),
    ),
    'SqueezeNet': ModelConfig(
        torchvision.models.squeezenet1_0,
        weights=torchvision.models.SqueezeNet1_0_Weights.IMAGENET1K_V1,
        default_traces=('features.1', 'features.3', 'features.4', 'features.5', 'features.7', 'features.8',
                        'features.9', 'features.10', 'features.12', 'classifier.2'),
    ),
    'InceptionV3': ModelConfig(
        torchvision.models.inception_v3,
        weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1,
        default_traces=('Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3', 'maxpool1', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3',
                        'maxpool2', 'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d',
                        'Mixed_6e', 'AuxLogits', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c', 'avgpool', 'dropout', 'fc'),
    ),
}

imagenet_quant_pretrained = {
    'ResNet50': QuantModelConfig(
        torchvision.models.quantization.resnet50,
        weights=torchvision.models.quantization.ResNet50_QuantizedWeights.IMAGENET1K_FBGEMM_V1,
    ),
    'GoogLeNet': QuantModelConfig(
        torchvision.models.quantization.googlenet,
        weights=torchvision.models.quantization.GoogLeNet_QuantizedWeights.IMAGENET1K_FBGEMM_V1,
    ),
    'InceptionV3': QuantModelConfig(
        torchvision.models.quantization.inception_v3,
        weights=torchvision.models.quantization.Inception_V3_QuantizedWeights.IMAGENET1K_FBGEMM_V1,
    ),
}

imagenet_clust_pretrained = {
    'ResNet18': ClustModelConfig(
        torchvision.models.quantization.resnet18,
        weights=torch.load(os.path.join('C:/', 'torch_data', 'clustered_weights', 'model_dict_resnet18.pt')),
        default_weights=torchvision.models.quantization.ResNet18_QuantizedWeights.IMAGENET1K_FBGEMM_V1,
    ),
    'GoogLeNet': ClustModelConfig(
        torchvision.models.quantization.googlenet,
        weights=torch.load(os.path.join('C:/', 'torch_data', 'clustered_weights', 'model_dict_googlenet.pt')),
        default_weights=torchvision.models.quantization.GoogLeNet_QuantizedWeights.IMAGENET1K_FBGEMM_V1,
    ),
}

imagenet_pruned = {
    'AlexNet': ChkpointModelConfig(
        torchvision.models.alexnet,
        chkpoint=os.path.join('C:/', 'torch_data', 'pruned_weights', 'AlexNet.pth'),
        weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1.IMAGENET1K_V1,
    ),
}