import argparse

import torch

from networks.cil_network import CIL_Net


class WeightAlignNet(CIL_Net):
    def __init__(self, network_name, pretrained, remove_existing_head, apply_to_bias):
        super().__init__(network_name, pretrained, remove_existing_head)
        self.apply_to_bias = apply_to_bias

    @staticmethod
    def extra_parser(args):
        parser = argparse.ArgumentParser()
        parser.add_argument("--apply-to-bias", action="store_true", help="Apply Weight align to bias term")
        return parser.parse_known_args(args)

    @torch.no_grad()
    def weight_align(self, old_classes, new_classes):
        """Implementation adjusted for repetition"""
        weights = self.head.weight.data
        old_classes = torch.tensor(old_classes)
        new_classes = torch.tensor(new_classes)
        oldnorm = torch.norm(weights[old_classes, :], p=2, dim=1)
        newnorm = torch.norm(weights[new_classes, :], p=2, dim=1)
        gamma = torch.mean(oldnorm) / torch.mean(newnorm)
        # Apply gamma to both weights and bias!
        self.head.weight.data[new_classes, :] *= gamma
        if self.apply_to_bias:
            self.head.bias.data[new_classes] *= gamma

    @torch.no_grad()
    def clip_head_weights_positive(self):
        self.head.weight.data[self.head.weight.data < 0] = 0
