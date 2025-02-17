from typing import List


import numpy as np
from argparse import ArgumentParser

import seaborn
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset

from datasets.data_loader import get_loaders


def _calculate_class_index_mapping_and_num_samples(loader: DataLoader, num_classes):
    class_index_map = {c: [] for c in range(num_classes)}
    for i, y in enumerate(loader.dataset.labels):
        class_index_map[y.item()].append(i)
    num_samples_per_class = np.zeros((num_classes))
    for key in class_index_map:
        num_samples_per_class[key] = len(class_index_map[key])

    return class_index_map, num_samples_per_class


class Manage_Loaders:
    """ Base class for managing the loaders. This is inherited by the specific scenario class. """

    def __init__(self, datasets, validation, num_tasks, batch_size, num_workers, pin_memory):
        self.datasets = datasets
        self.validation = validation
        self.num_tasks = num_tasks
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Keep num tasks to 1 here we want to extract all the data into train validation and test then split it up further
        self.total_trn_loader, self.total_val_loader, self.total_tst_loader, taskcla = get_loaders(self.datasets, 1, None,
                                                                                                   self.batch_size, self.num_workers,
                                                                                                   self.pin_memory, validation=self.validation)
        # Only
        self.total_trn_loader = self.total_trn_loader[0]
        self.total_val_loader = self.total_val_loader[0]
        self.total_tst_loader = self.total_tst_loader[0]
        self.num_classes = taskcla[0][1]

        self.class_index_map_train, self.num_samples_map_train = _calculate_class_index_mapping_and_num_samples(self.total_trn_loader, self.num_classes)
        self.class_index_map_val, self.num_samples_map_val = _calculate_class_index_mapping_and_num_samples(self.total_val_loader, self.num_classes)
        self.class_index_map_test, self.num_samples_map_test = _calculate_class_index_mapping_and_num_samples(self.total_tst_loader, self.num_classes)

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        return parser.parse_known_args(args)

    def _calculate_class_mapping(self):
        self.inverse_class_mapping = {}
        cur_cls_max = 0

        known_classes = set()
        for cur_task in range(self.num_tasks):
            for cur_cls in range(self.num_classes):
                if self._scenario_samples[cur_cls, cur_task] > 0:
                    if cur_cls not in known_classes:
                        self.inverse_class_mapping[cur_cls_max] = cur_cls
                        known_classes.add(cur_cls)
                        cur_cls_max += 1

        self.class_mapping = {self.inverse_class_mapping[k]: k for k in self.inverse_class_mapping}
        assert cur_cls_max == self.num_classes, "Not all classes trained on? check class mapping and scenario_table"

    def _apply_class_mapping_on_dataset(self):
        # Apply new Class mappings we can now ensure that the number of classes is growing
        def apply_mapping(dataset: Dataset):
            if isinstance(dataset, ConcatDataset):
                for ds in dataset.datasets:
                    apply_mapping(ds)
            elif isinstance(dataset, Subset):
                apply_mapping(dataset.dataset)
            else:
                return dataset.labels.apply_(lambda x: self.class_mapping[x])

        apply_mapping(self.total_trn_loader.dataset)
        apply_mapping(self.total_val_loader.dataset)
        apply_mapping(self.total_tst_loader.dataset)

    def get_classes_present_in_t(self, t) -> List[int]:
        assert 0 <= t < self.num_tasks
        classes_present = np.nonzero(np.squeeze(self._scenario_samples[:, t]))
        return classes_present[0].tolist()

    def get_classes_present_so_far(self, t) -> List[int]:
        assert 0 <= t < self.num_tasks
        classes_present = np.nonzero(np.squeeze(np.sum(self._scenario_samples[:, :(t+1)], axis=1)))
        return classes_present[0].tolist()

    def get_trn_loader(self, t):
        assert 0 <= t < self.num_tasks
        return self._list_trn_loaders[t]

    def get_val_loader(self, t):
        assert 0 <= t < self.num_tasks
        classes_present_preorder = self.get_classes_present_in_t(t)
        indices = []
        for c in classes_present_preorder:
            indices += self.class_index_map_val[c]
        test_ds = Subset(self.total_val_loader.dataset, indices)
        return DataLoader(test_ds, self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def get_accumulated_val_loader(self, t):
        assert 0 <= t < self.num_tasks
        classes_present_preorder = self.get_classes_present_so_far(t)
        indices = []
        for c in classes_present_preorder:
            indices += self.class_index_map_val[c]
        test_ds = Subset(self.total_val_loader.dataset, indices)
        return DataLoader(test_ds, self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def get_tst_loader(self, t):
        """Gets the testloader for the iteration """
        assert 0 <= t < self.num_tasks
        classes_present_preorder = self.get_classes_present_so_far(t)
        indices = []
        for c in classes_present_preorder:
            indices += self.class_index_map_test[c]

        test_ds = Subset(self.total_tst_loader.dataset, indices)
        return DataLoader(test_ds, self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def get_tst_loader_for_classes_only_in_t(self, t):
        """Gets the testloader for the iteration """
        assert 0 <= t < self.num_tasks
        classes_present_preorder = np.nonzero(np.squeeze(self._scenario_samples[:, t]))[0].tolist()
        indices = []
        for c in classes_present_preorder:
            indices += self.class_index_map_test[c]

        test_ds = Subset(self.total_tst_loader.dataset, indices)
        return DataLoader(test_ds, self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def split_loaders(self):
        task_loaders = []
        self._scenario_samples = self._generate_tasks()
        # self.plot_config(scenario_samples)

        # calculate new class order! -> remap classes to a continual list so that modifying head size works!
        # real growing classes
        self._calculate_class_mapping()
        self._apply_class_mapping_on_dataset()

        # Create loaders from random elements
        for i in range(self.num_tasks):
            subset_idcs = []
            for c_idx in range(self.num_classes):
                samples = self._scenario_samples[c_idx, i]
                if samples > 0:
                    result = self.class_index_map_train[c_idx]
                    np.random.shuffle(result)
                    result = result[:samples]
                    subset_idcs += result

            exp_dataset = Subset(self.total_trn_loader.dataset, subset_idcs)
            exp_loader = DataLoader(exp_dataset, pin_memory=self.pin_memory, batch_size=self.batch_size,
                                    shuffle=True, num_workers=self.num_workers)
            task_loaders.append(exp_loader)
        self._list_trn_loaders = task_loaders

    def plot_config(self, scenario_table):
        fig = plt.figure(figsize=(9.6, 5.9))
        # plt.title("Class Table")
        target_width = 1920
        target_height = 1080
        color_classes = (np.array(seaborn.color_palette(n_colors=self.num_classes)) * 255).astype(int)
        np.random.shuffle(color_classes)
        colormapped_scenario_table = np.ones((scenario_table.shape[0], scenario_table.shape[1], 3), dtype=np.uint8) * 255
        for i in range(scenario_table.shape[0]):
            for j in range(scenario_table.shape[1]):

                if scenario_table[i, j] > 0:
                    colormapped_scenario_table[i, j, :] = color_classes[i]

        plt.imshow(colormapped_scenario_table, interpolation="nearest", extent=[0, target_width, 0, target_height])
        plt.xlabel("Task / Experience")
        plt.ylabel("Class")

        start = (target_width / scenario_table.shape[1]) // 2
        labels_x = np.arange(scenario_table.shape[1]) if scenario_table.shape[1] < 20 else [f"{i}" if i % 10 == 0 else "" for i in range(scenario_table.shape[1])]
        plt.xticks(np.arange(start, target_width, target_width / scenario_table.shape[1]), labels_x)
        start = (target_height / scenario_table.shape[0]) // 2
        plt.yticks(np.arange(start, target_height, target_height / (scenario_table.shape[0])), [""] * scenario_table.shape[0])
        plt.savefig("scenario_uniform.png")
        plt.show()

    def _generate_tasks(self):
        raise NotImplementedError("Use this base class for inheritence and fill generate tasks")
