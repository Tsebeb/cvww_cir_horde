import copy
from typing import List

import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler, DataLoader


class ClassPrototype:
    def __init__(self, class_id: int, mean: List[np.ndarray], std: List[np.ndarray], fe_idx_list: List[int]):
        self.class_id = class_id
        self.mean = mean
        self.std = std
        self.fe_idx_list = fe_idx_list

        assert len(mean) == len(std)
        self.total_length = 0
        for m, s in zip(mean, std):
            assert len(m) == len(s)
            self.total_length += len(m)

    def get_total_mean(self):
        mean = np.zeros(self.total_length)
        length_sofar = 0
        for m in self.mean:
            size_cur = len(m)
            mean[length_sofar: length_sofar + size_cur] = m[:]
            length_sofar += size_cur
        return mean

    def get_total_std(self):
        std = np.zeros(self.total_length)
        length_sofar = 0
        for s in self.std:
            size_cur = len(s)
            std[length_sofar: length_sofar + size_cur] = s[:]
            length_sofar += size_cur
        return std

    def create_sub_class_prototype(self, new_fe_idx_list):
        mean_subset = []
        std_subset = []

        for i in new_fe_idx_list:
            list_idx = self.fe_idx_list.index(i)
            mean_subset.append(self.mean[list_idx])
            std_subset.append(self.std[list_idx])

        return ClassPrototype(self.class_id, mean_subset, std_subset, copy.deepcopy(new_fe_idx_list))

    def extend_class_prototype(self, additional_mean, additional_std, additional_fe_idx):
        self.mean.append(additional_mean)
        self.std.append(additional_std)
        self.fe_idx_list.append(additional_fe_idx)
        self.total_length += len(additional_mean)

    def get_common_class_list(self, other_prototype):
        common_fes = []
        for fe_idx in self.fe_idx_list:
            if fe_idx in other_prototype.fe_idx_list:
                common_fes.append(fe_idx)
        return common_fes

    def calculate_distribution_overlap(self, other_prototype):
        assert self.total_length == other_prototype.total_length
        for a, b in zip(self.fe_idx_list, other_prototype.fe_idx_list):
            assert a == b, "comparing embeddings of different feature extractors!"
        mu_1, std_1 = self.get_total_mean(), self.get_total_std()
        mu_2, std_2 = other_prototype.get_total_mean(), other_prototype.get_total_std()
        var_1, var_2 = std_1 ** 2, std_2 ** 2

        avg_var = (var_1 + var_2) / 2.0
        det_var1 = np.power(np.prod(var_1), 0.25)
        det_var2 = np.power(np.prod(var_2), 0.25)
        inverse_var_arg = 1 / (avg_var)
        det_avg_var = np.power(np.prod(avg_var), 0.5)
        mu_dif = mu_1 - mu_2
        return det_var1 * det_var2 / det_avg_var * np.exp(-1. / 8. * np.sum(mu_dif * inverse_var_arg * mu_dif))


def convert_class_prototypes(initial_mean_embedding, initial_std_embedding, mean_embedding, std_embedding,
                             initial_max_samples_seen, max_samples_seen) -> List[ClassPrototype]:
    classes = initial_max_samples_seen.size(0)
    converted_prototypes = []
    for c in range(classes):
        c_mean, c_std, c_fe_idx = [], [], []
        if initial_max_samples_seen[c] > 0:
            c_mean.append(initial_mean_embedding[c, :].cpu().numpy())
            c_std.append(initial_std_embedding[c, :].cpu().numpy())
            c_fe_idx.append(0)

        for fe_idx in range(max_samples_seen.size(1)):
            if max_samples_seen[c, fe_idx] > 0:
                c_mean.append(mean_embedding[c, fe_idx, :].cpu().numpy())
                c_std.append(std_embedding[c, fe_idx, :].cpu().numpy())
                c_fe_idx.append(1 + fe_idx)

        if c_fe_idx:
            converted_prototypes.append(ClassPrototype(c, c_mean, c_std, c_fe_idx))
    return converted_prototypes


def convert_class_prototypes_without_init(mean_embedding, std_embedding, max_samples_seen) -> List[ClassPrototype]:
    classes = mean_embedding.size(0)
    converted_prototypes = []
    for c in range(classes):
        c_mean, c_std, c_fe_idx = [], [], []
        for fe_idx in range(max_samples_seen.size(1)):
            if max_samples_seen[c, fe_idx] > 0:
                c_mean.append(mean_embedding[c, fe_idx, :].cpu().numpy())
                c_std.append(std_embedding[c, fe_idx, :].cpu().numpy())
                c_fe_idx.append(fe_idx)

        if c_fe_idx:
            converted_prototypes.append(ClassPrototype(c, c_mean, c_std, c_fe_idx))
    return converted_prototypes


def calculate_class_prototypes(dataloader: DataLoader, modules: list, device, class_idx_map) -> List[ClassPrototype]:
    class_prototypes = {cls_idx: {"mean": [], "std": [], "idcs": []} for cls_idx in class_idx_map}
    for i, m in enumerate(modules):
        class_parts = calculate_partial_class_prototype(dataloader, m, device, class_idx_map)
        for c_idx in class_parts:
            class_prototypes[c_idx]["mean"].append(class_parts[c_idx]["mean"])
            class_prototypes[c_idx]["std"].append(class_parts[c_idx]["std"])
            class_prototypes[c_idx]["idcs"].append(i)

    prototypes = [ClassPrototype(c, class_prototypes[c]["mean"], class_prototypes[c]["std"], class_prototypes[c]["idcs"]) for c in class_prototypes]
    return prototypes


def calculate_partial_class_prototype(dataloader: DataLoader, feature_extractor, device, class_idx_map):
    with torch.inference_mode():
        feature_extractor.eval()
        cls_parts = {}
        for cls_idx in class_idx_map:
            cls_parts[cls_idx] = {}
            cls_sampler = SubsetRandomSampler(indices=class_idx_map[cls_idx])
            cls_loader = DataLoader(dataloader.dataset, batch_size=dataloader.batch_size,
                                    shuffle=False, sampler=cls_sampler)
            cls_features = None
            for image, target in cls_loader:
                image = image.to(device)
                features = feature_extractor(image)

                cls_features = features if cls_features is None else torch.vstack((cls_features, features))

            cls_parts[cls_idx]["mean"] = torch.mean(cls_features, dim=0).cpu().numpy()
            cls_parts[cls_idx]["std"] = torch.std(cls_features, dim=0).cpu().numpy()
    return cls_parts
