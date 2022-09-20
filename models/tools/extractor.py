import os
from typing import Callable

import numpy as np
import torch
from torch.fx import Interpreter


AUTO = 'auto'


def weight_trace(param_name):
    return 'weight' in param_name

def bias_trace(param_name):
    return 'bias' in param_name

def contains_trace(substring):
    def tfunc(param_name):
        return substring in param_name
    return tfunc


class ModelExtractor(object):
    def __init__(self, target_model: torch.nn.Module=None, output_modelname: str='model', device: str=AUTO, verbose=False):
        self.target_model = target_model          # target model (torch.nn.Module)
        self.output_modelname = output_modelname  # name of saved model
        self._params = {}      # extracted parameters
        self._activation = {}  # extracted output activations
        self._hook_names = []  # name of registered hooks
        self._traces = []      # parameter traces
        self._verbose = verbose
        self.device = device   # defined target device (mainly 'cpu' or 'cuda')

        if self.device == AUTO:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def reset(self):
        self._params = {}
        self._activation = {}
        self._hook_names = []
        self._traces = []

    def add_param_trace(self, trace: Callable):
        if trace not in self._traces:
            self._traces.append(trace)

    def remove_param_trace(self, trace: Callable):
        if trace in self._traces:
            self._traces.remove(trace)

    def register_hook(self, layer: torch.nn.Module, name: str=AUTO):
        if name == AUTO:
            hidx = 0
            while f"hook{hidx}" in self._hook_names: hidx += 1
            name = f"hook{hidx}"
            self._hook_names.append(name)
        layer.register_forward_hook(self._extract_hook(name))

    def register_hook_by_type(self, layer_type):
        for layer in list(filter(lambda m: type(m) == layer_type, self.target_model.modules())):
            hidx = 0
            while f"{type(layer).__name__}_{hidx}" in self._hook_names: hidx += 1
            name = f"{type(layer).__name__}_{hidx}"
            self._hook_names.append(name)
            layer.register_forward_hook(self._extract_hook(name))

    def _extract_hook(self, name: str):
        def hook(model, layer_input, layer_output):
            oidx = 0
            while f"{name}_output{oidx}" in self._activation.keys(): oidx += 1
            save_output_name = f"{name}_output{oidx}"
            self._activation[save_output_name] = layer_output.contiguous().detach()
            if self._verbose:
                print(f'extracting {save_output_name} (type: {self._activation[save_output_name].type()})')

        return hook

    def extract_activation(self, dataloader, max_iter=5):
        iter_cnt = 0
        self.target_model.eval()

        for X, y in dataloader:
            if iter_cnt >= max_iter:
                break
            else:
                iter_cnt += 1
            X, y = X.to(self.device), y.to(self.device)
            self.target_model(X)

    def extract_params(self):
        for param_name in self.target_model.state_dict():
            for trace in self._traces:
                if trace(param_name):
                    parsed_name = f"{self.output_modelname}_{param_name.replace('.', '_')}"
                    try:
                        if self._verbose:
                            print(f"extracting {parsed_name}")
                        self._params[parsed_name] = self.target_model.state_dict()[param_name].detach()
                    except:
                        print(f"error occurred on extracting {parsed_name}")
                    break

    def save_params(self, savepath: str=AUTO, only_quant=False):
        if savepath == AUTO:
            savepath = os.path.join(os.curdir, 'extracted_output', self.output_modelname, 'params')
        os.makedirs(savepath, exist_ok=True)

        saved_params = []

        for param_name in self._params.keys():
            barr = self._params[param_name].detach()
            if self.device != 'cpu': barr = barr.to('cpu')
            if barr.dtype in (torch.qint8, torch.qint32, torch.quint8):
                barr = barr.int_repr()
            elif only_quant:
                continue
            with open(os.path.join(savepath, f"{param_name}"), 'wb') as file:
                saved_params.append(param_name)
                file.write(barr.numpy().tobytes())

        with open(os.path.join(savepath, 'filelist.txt'), 'wt') as filelist:
            filelist.write('\n'.join([os.path.join(savepath, layer_name) for layer_name in saved_params]))

    def save_activation(self, savepath: str=AUTO, only_quant=False):
        if savepath == AUTO:
            savepath = os.path.join(os.curdir, 'extracted_output', self.output_modelname, 'activations')
        os.makedirs(savepath, exist_ok=True)

        saved_acts = []

        for layer_name in self._activation.keys():
            barr = self._activation[layer_name].detach()
            if self.device != 'cpu': barr = barr.to('cpu')
            if barr.dtype in (torch.qint8, torch.qint32, torch.quint8):
                barr = barr.int_repr()
            elif only_quant:
                continue
            with open(os.path.join(savepath, f"{layer_name}"), 'wb') as file:
                saved_acts.append(layer_name)
                file.write(barr.numpy().tobytes())

        with open(os.path.join(savepath, 'filelist.txt'), 'wt') as filelist:
            filelist.write('\n'.join([os.path.join(savepath, layer_name) for layer_name in saved_acts]))


class QuantModelExtractor(Interpreter):
    def __init__(self, target_model, output_modelname='model', verbose=False):
        super(QuantModelExtractor, self).__init__(target_model)
        self.target_model = target_model          # target model (fx graph model)
        self.output_modelname = output_modelname  # name of saved model
        self._params = {}      # extracted parameters
        self._activation = {}  # extracted output activations
        self._traces = []      # parameter traces
        self._verbose = verbose
        self.device = 'cpu'

    def reset(self):
        self._params = {}
        self._activation = {}
        self._traces = []

    def add_param_trace(self, trace: Callable):
        if trace not in self._traces:
            self._traces.append(trace)

    def remove_param_trace(self, trace: Callable):
        if trace in self._traces:
            self._traces.remove(trace)

    def call_module(self, target, *args, **kwargs):
        for kw in self.traces:
            if kw in target.split('.'):
                idx = 0
                save_output_name = f"{self.output_modelname}_{target}_output{idx}"
                if save_output_name in self.features:
                    idx += 1
                    save_output_name = f"{self.output_modelname}_{target}_output{idx}"

                if self._verbose:
                    print(f'extracting {save_output_name}')
                self.features[save_output_name] = super().call_module(target, *args, **kwargs).int_repr().detach()
        return super().call_module(target, *args, **kwargs)

    def extract_activation(self, dataloader, max_iter=5):
        iter_cnt = 0

        for X, y in dataloader:
            if iter_cnt > max_iter:
                break
            else:
                iter_cnt += 1
            X, y = X.to(self.device), y.to(self.device)
            self.run(X)

    def extract_params(self):
        for param_name in self.target_model.state_dict():
            for trace in self._traces:
                if trace(param_name):
                    parsed_name = f"{self.output_modelname}_{param_name.replace('.', '_')}"
                    try:
                        if self._verbose:
                            print(f"extracting {parsed_name}")
                        self._params[parsed_name] = self.target_model.state_dict()[param_name].detach().cpu().int_repr()
                    except Exception as error:
                        print(f"error occurred on extracting {parsed_name} ({error}")
                    break

    def save_params(self, savepath: str=AUTO):
        if savepath == AUTO:
            savepath = os.path.join(os.curdir, 'extracted_output', self.output_modelname, 'params')
        os.makedirs(savepath, exist_ok=True)

        for param_name in self._params.keys():
            barr = self._params[param_name].detach().numpy()
            with open(os.path.join(savepath, f"{param_name}"), 'wb') as file:
                file.write(barr)

        with open(os.path.join(savepath, 'filelist.txt'), 'wt') as filelist:
            filelist.write('\n'.join([os.path.join(savepath, layer_name) for layer_name in self._params.keys()]))

    def save_activation(self, savepath: str=AUTO):
        if savepath == AUTO:
            savepath = os.path.join(os.curdir, 'extracted_output', self.output_modelname, 'activations')
        os.makedirs(savepath, exist_ok=True)

        for layer_name in self._activation.keys():
            barr = self._activation[layer_name].detach().numpy()
            with open(os.path.join(savepath, f"{layer_name}"), 'wb') as file:
                file.write(barr)

        with open(os.path.join(savepath, 'filelist.txt'), 'wt') as filelist:
            filelist.write('\n'.join([os.path.join(savepath, layer_name) for layer_name in self._activation.keys()]))