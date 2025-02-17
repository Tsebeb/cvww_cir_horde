import argparse
from copy import deepcopy

import torch
from torch.nn import ModuleList, Linear

from networks.cil_network import CIL_Net
from networks.utils import CosineLinear


class InitHordeModel(CIL_Net):
    def __init__(self, network_name, pretrained, remove_existing_head, initial_network_name,
                 num_fe, head_latent_dim_size, head_dropout, head_activation_fct):
        self.initial_network_name = initial_network_name if initial_network_name is not None else network_name
        self.network_name = network_name
        self.pretrained = pretrained
        self.remove_existing_head = remove_existing_head
        super(InitHordeModel, self).__init__(network_name, pretrained, remove_existing_head)
        self.number_of_feature_extractors = num_fe - 1 # for initial model
        self.initial_feature_extractor = None
        self.feature_extractors = ModuleList()
        del self.model  # do not need the base model now
        self.hidden_dim_size = head_latent_dim_size
        self.dropout = head_dropout
        self.activation = head_activation_fct
        self.linear = None
        self.__old_size = 0

        # test initial init feature
        _, _, self.init_out_size = self._create_base_model(self.initial_network_name, self.pretrained, self.remove_existing_head)

    def modify_head(self, num_outputs):
        if self.linear is None:
            if self.hidden_dim_size <= 0:
                self.linear = self.head_class(self.init_out_size + self.out_size * self.number_of_feature_extractors, num_outputs)
            else:
                layers = [Linear(self.init_out_size + self.out_size * self.number_of_feature_extractors, self.hidden_dim_size)]
                if self.activation == "relu":
                    layers.append(torch.nn.ReLU())
                else:
                    raise RuntimeError("Unsupported Activation add here")
                if self.dropout > 0:
                    layers.append(torch.nn.Dropout(self.dropout))
                layers.append(self.head_class(self.hidden_dim_size, num_outputs))
                self.linear = torch.nn.Sequential(*layers)
        else:
            if self.hidden_dim_size <= 0:
                new_linear = self.head_class(self.init_out_size + self.out_size * self.number_of_feature_extractors, num_outputs)
                if self.head_class == Linear:
                    new_linear.weight.data[:self.__old_size, :] = self.linear.weight.data[:, :]
                    new_linear.bias.data[:self.__old_size] = self.linear.bias.data[:]
                elif self.head_class == CosineLinear:
                    new_linear.weight.data[:self.__old_size, :] = self.linear.weight.data[:, :]
                    new_linear.sigma.data[:] = self.linear.sigma.data[:]
                else:
                    raise RuntimeError("unsupported weight copying of head type please implement rule here")
                self.linear = new_linear
            else:
                new_linear = self.head_class(self.init_out_size + self.out_size * self.number_of_feature_extractors, num_outputs)
                if self.head_class == Linear:
                    new_linear.weight.data[:, :self.__old_size] = self.linear[-1].weight.data[:, :]
                    new_linear.bias.data[:self.__old_size] = self.linear[-1].bias.data[:]
                elif self.head_class == CosineLinear:
                    new_linear.weight.data[:self.__old_size, :] = self.linear[-1].weight.data[:, :]
                    new_linear.sigma.data[:] = self.linear[-1].sigma.data[:]
                else:
                    raise RuntimeError("unsupported weight copying of head type please implement rule here")
                self.linear[-1] = new_linear
        self.__old_size = num_outputs

    @staticmethod
    def extra_parser(args):
        parser = argparse.ArgumentParser()
        parser.add_argument("--initial-network-name", type=str, default=None, required=False)
        parser.add_argument('--num-fe', type=int, default=5, required=False,
                            help='Number of feature extractors used for the backbone of the ')
        parser.add_argument("--head-latent-dim-size", type=int, default=-1,
                            help="The number of the latent dimension between the output head and the feature extractors. if the size is negative or 0 then the entire head is "
                                 "a single Linear layer no intermediate hidden layer")
        parser.add_argument("--head-activation-fct", type=str, default="relu")
        parser.add_argument("--head-dropout", type=float, default=0.0, help="The dropout for the head")
        return parser.parse_known_args(args)

    def has_room_to_grow(self):
        return len(self.feature_extractors) < self.number_of_feature_extractors

    def forward(self, x: torch.Tensor, return_features=False):
        features = torch.randn((x.size(0), self.init_out_size + self.number_of_feature_extractors * self.out_size), device=x.device)
        features = features * 0.01
        features[:, :self.init_out_size] = self.initial_feature_extractor(x)
        for m, model in enumerate(self.feature_extractors):
            features[:, self.init_out_size + m * self.out_size : self.init_out_size + (m + 1) * self.out_size] = model(x)
        output = self.linear(features)

        if return_features:
            return output, features
        return output

    def get_total_features(self, x):
        features = torch.randn((x.size(0), self.init_out_size + self.number_of_feature_extractors * self.out_size), device=x.device)
        features = features * 0.01
        features[:, :self.init_out_size] = self.initial_feature_extractor(x)
        for m, model in enumerate(self.feature_extractors):
            features[:, self.init_out_size + m * self.out_size : self.init_out_size + (m + 1) * self.out_size] = model(x)
        return features

    def _freeze_feature_extractor(self, fe: torch.nn.Module):
        fe.eval()
        for param in fe.parameters():
            param.requires_grad = False
        return fe

    def add_feature_extractor(self, feature_extractor: torch.nn.Module, position: int):
        feature_extractor = self._freeze_feature_extractor(feature_extractor)
        # Append FE
        if len(self.feature_extractors) < self.number_of_feature_extractors:
            self.feature_extractors.append(feature_extractor)
        else:
            self.feature_extractors[position] = feature_extractor

    def add_initial_feature_extractor(self, initial_feature_extractor: torch.nn.Module):
        initial_feature_extractor = self._freeze_feature_extractor(initial_feature_extractor)
        self.initial_feature_extractor = initial_feature_extractor

    def train(self, mode: bool = True):
        # Keep Feature Extractors frozen!
        self.linear.train(mode)

    def get_initial_model(self):
        model, _, _ = self._create_base_model(self.initial_network_name, self.pretrained, self.remove_existing_head)
        return model

    def get_fe_model(self):
        model, _, _ = self._create_base_model(self.network_name, self.pretrained, self.remove_existing_head)
        return model

