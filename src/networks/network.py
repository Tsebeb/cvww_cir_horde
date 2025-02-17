import copy

import torch
from torch import nn
from copy import deepcopy

from networks.resnet18_lucir import CosineLinear


class LLL_Net(nn.Module):
    """Basic class for implementing networks"""

    def __init__(self, model, remove_existing_head=False):
        head_var = model.head_var
        assert type(head_var) == str
        assert not remove_existing_head or hasattr(model, head_var), \
            "Given model does not have a variable called {}".format(head_var)
        assert not remove_existing_head or type(getattr(model, head_var)) in [nn.Sequential, nn.Linear], \
            "Given model's head {} does is not an instance of nn.Sequential or nn.Linear".format(head_var)
        super(LLL_Net, self).__init__()

        self.model = model
        last_layer = getattr(self.model, head_var)
        self.cosine_head = isinstance(last_layer[-1], CosineLinear) if type(last_layer) == nn.Sequential else isinstance(last_layer, CosineLinear)

        if remove_existing_head:
            if type(last_layer) == nn.Sequential:
                self.out_size = last_layer[-1].in_features
                # strips off last linear layer of classifier
                del last_layer[-1]
            elif type(last_layer) == nn.Linear:
                self.out_size = last_layer.in_features
                # converts last layer into identity
                # setattr(self.model, head_var, nn.Identity())
                # WARNING: this is for when pytorch version is <1.2
                setattr(self.model, head_var, nn.Sequential())
        else:
            self.out_size = last_layer.out_features

        self.head = None
        self.cls_progression = []
        self._initialize_weights()

    def modify_head(self, num_outputs):
        """Add a new head with the corresponding number of outputs. Also update the number of classes per task and the
        corresponding offsets
        """
        if self.head is None:
            self.head = nn.Linear(self.out_size, num_outputs) if not self.cosine_head else CosineLinear(self.out_size, num_outputs)
        else:
            old_head = copy.deepcopy(self.head)
            # new output shape and size
            self.head = nn.Linear(self.out_size, num_outputs) if not self.cosine_head else CosineLinear(self.out_size, num_outputs)
            # copy old heads weights and biases
            old_size = old_head.out_features
            self.head.weight.data[:, :old_size] = old_head.weight.data
            self.head.bias.data[:old_size] = old_head.bias.data

        # we re-compute instead of append in case an approach makes changes to the heads
        self.cls_progression.append(num_outputs)

    def forward(self, x, return_features=False):
        """Applies the forward pass

        Simplification to work on multi-head only -- returns all head outputs in a list
        Args:
            x (tensor): input images
            return_features (bool): return the representations before the heads
        """
        x = self.model(x)
        assert self.head is not None, "Cannot access any head"
        y = self.head(x)
        if return_features:
            return y, x
        else:
            return y

    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
        return

    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False

    def freeze_backbone(self):
        """Freeze all parameters from the main model, but not the heads"""
        for param in self.model.parameters():
            param.requires_grad = False

    def freeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _initialize_weights(self):
        """Initialize weights using different strategies"""
        # TODO: add different initialization strategies
        pass
