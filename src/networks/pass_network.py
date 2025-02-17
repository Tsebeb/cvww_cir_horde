import argparse
import copy

import torch.nn

from networks.cil_network import CIL_Net


class PASS_Net(CIL_Net):
    def __init__(self, network_name, pretrained, remove_existing_head):
        super().__init__(network_name, pretrained, remove_existing_head)
        #
        self.orientation_class_head = None
        self._orientation_scale_factor = 4

    @staticmethod
    def extra_parser(args):
        parser = argparse.ArgumentParser()
        return parser.parse_known_args(args)

    def modify_head(self, num_outputs):
        # Adjusts the normal head
        super().modify_head(num_outputs)

        # Adjust the head with orientation output as well as the original head
        if self.orientation_class_head is None:
            self.orientation_class_head = self.head_class(self.out_size, num_outputs * self._orientation_scale_factor)
        else:
            new_orientation_head = self.head_class(self.out_size, num_outputs * self._orientation_scale_factor)
            # copy old heads weights and biases
            old_size = self.orientation_class_head.out_features
            new_orientation_head.weight.data[:old_size, :] = self.orientation_class_head.weight.data[:, :]
            new_orientation_head.bias.data[:old_size] = self.orientation_class_head.bias.data[:]
            self.orientation_class_head = new_orientation_head

    def orientation_forward(self, x, return_features=False):
        features = self.model(x)
        y = self.orientation_class_head(features)
        if return_features:
            return y, features
        return y

    def collapse_orientation_head(self):
        """Reduces the size of the head. Needed for reducing from the
        in the original code this method was called saveOptions()
        """
        assert self.head.out_features * self._orientation_scale_factor == self.orientation_class_head.out_features
        self.head.weight.data[:, :] = self.orientation_class_head.weight.data[::self._orientation_scale_factor, :]
        self.head.bias.data[:] = self.orientation_class_head.bias.data[::self._orientation_scale_factor]


