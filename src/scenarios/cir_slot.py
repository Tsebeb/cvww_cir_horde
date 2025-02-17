import argparse
import random

import numpy as np
from argparse import ArgumentParser

from torch.utils.data import DataLoader, Subset
from scenarios.manage_loaders import Manage_Loaders


class Scenario_Loader(Manage_Loaders):
    """ This class implements the scenario as it is proposed in CIR for the CLVISION Challenge 2023 @ CVPR
         - this means each class has a initial geometric discovery probability
         - the reoccurence of the classes are fixed

    """

    def __init__(self, datasets, validation, num_tasks, batch_size, num_workers, pin_memory, slots_k: int, shuffle_idcs: bool):
        self.slots_k = slots_k
        self.shuffle_idcs = shuffle_idcs

        assert self.slots_k <= num_tasks
        assert self.num_classes % num_tasks == 0
        assert (self.num_tasks * self.slots_k) % self.num_classes == 0
        super(Scenario_Loader, self).__init__(datasets, validation, num_tasks, batch_size, num_workers, pin_memory)

    @staticmethod
    def extra_parser(args):
        parser = argparse.ArgumentParser()
        parser.add_argument("--slots-k", type=int, help="")
        parser.add_argument("--shuffle-idcs", action="store_true")
        return parser.parse_known_args(args)

    def split_loaders(self):
        task_loaders = []

        slots = self._generate_tasks()
        for t in range(self.num_tasks):
            total_idcs = []

            available_classes = slots.keys()
            selected_classes = np.random.choice(available_classes, self.slots_k, replace=False)

            for class_idx in selected_classes:
                selected_slot_idx = np.random.choice(range(len(slots[class_idx])), 1, replace=False)
                total_idcs += slots[class_idx].pop(selected_slot_idx)

            task_ds = Subset(self.total_trn_loader, total_idcs)
            task_dl = DataLoader(task_ds, batch_size=self.batch_size,
                                 num_workers=self.num_workers, shuffle=True,
                                 pin_memory=self.pin_memory)
            task_loaders.append(task_dl)
        return task_loaders

    def _generate_tasks(self):
        slots = {}

        for class_idx in self.class_index_map_train.keys():
            slots[class_idx] = []
            ksample = int(len(self.class_index_map_train[class_idx]) / self.slots_k)

            base_idcs = self.class_index_map_train[class_idx] if not self.shuffle_idcs else random.shuffle(self.class_index_map_train[class_idx])
            for k in range(self.num_tasks * self.slots_k / self.num_classes):
                slots[class_idx].append(base_idcs[k * ksample : (k + 1) * ksample])
        return slots


