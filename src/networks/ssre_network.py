import argparse
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.cil_network import CIL_Net
from networks.resnet_18_ssre import resnet18_ssre

_allowed_networks = ["resnet18_ssre", "slimresnet18_ssre", "resnet18_ssre_bn", "slimresnet18_ssre_bn", "resnet18_ssre_bn_tiny"]
class SSRE_Net(CIL_Net):
    def __init__(self, network_name, pretrained, remove_existing_head):
        assert network_name in _allowed_networks
        self.network_name = network_name
        self.pretrained = pretrained
        self.remove_existing_head = remove_existing_head
        super().__init__(network_name, pretrained, remove_existing_head)
        self.model.switch("normal")

    @staticmethod
    def extra_parser(args):
        parser = argparse.ArgumentParser()

        return parser.parse_known_args(args)

    def network_expansion(self):
        for p in self.model.parameters():
            p.requires_grad = True
        for k, v in self.model.named_parameters():
            if 'adapter' not in k:
                v.requires_grad = False
                # self._network.convnet.re_init_params() # do not use!
        self.model.switch("parallel_adapters")

    def network_compression(self):
        model_dict = self.model.state_dict()
        for k, v in model_dict.items():
            if 'adapter' in k:
                k_conv3 = k.replace('adapter', 'conv')
                if 'weight' in k:
                    model_dict[k_conv3] = model_dict[k_conv3] + F.pad(v, [1, 1, 1, 1], 'constant', 0)
                    model_dict[k] = torch.zeros_like(v)
                elif 'bias' in k:
                    model_dict[k_conv3] = model_dict[k_conv3] + v
                    model_dict[k] = torch.zeros_like(v)
                else:
                    assert 0
        self.model.load_state_dict(model_dict)
        self.model.switch("normal")
