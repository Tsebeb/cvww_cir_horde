import argparse
import copy

import torch.nn

from networks.cil_network import CIL_Net


class IL2ANet(CIL_Net):
    def __init__(self, network_name, pretrained, remove_existing_head):
        super().__init__(network_name, pretrained, remove_existing_head)

    @staticmethod
    def extra_parser(args):
        parser = argparse.ArgumentParser()
        return parser.parse_known_args(args)

    def shrink_head(self, num_outputs):
        """Reduces the size of the head. Needed for reducing from the
        in the original code this method was called saveOptions()
        """
        if self.head is None:
            self.head = self.head_class(self.out_size, num_outputs)
        else:
            new_head = self.head_class(self.out_size, num_outputs)
            new_head.weight.data[:, :] = self.head.weight.data[:num_outputs, :]
            new_head.bias.data[:] = self.head.bias.data[:num_outputs]
            new_head.to(self.head.weight.device)
            self.head = new_head


