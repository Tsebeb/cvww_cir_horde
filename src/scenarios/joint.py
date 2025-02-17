import argparse
import random

import numpy as np

from scenarios.manage_loaders import Manage_Loaders


class Scenario_Loader(Manage_Loaders):
    """ This class implements the classical CIL setting specify the number of start classes and the rest of the classes will be divided
     with the rest of the available tasks. The remaining must be divisible by the num_tasks - 1.
     For example CIFAR 100 can have 10 starting classes and 10 num_tasks so that every task has exactly 10 classes.
     Alternatively We could do 50 starting classes and 11 num_tasks to have the 50 10x5 split.
     All the training data will be seen in """

    def __init__(self, datasets, validation, num_tasks, batch_size, num_workers, pin_memory):
        assert num_tasks == 1, "Joint Training only allows 1 task!"
        super(Scenario_Loader, self).__init__(datasets, validation, 1, batch_size, num_workers, pin_memory)

    @staticmethod
    def extra_parser(args):
        parser = argparse.ArgumentParser()
        return parser.parse_known_args(args)

    def _generate_tasks(self):
        n_samples_table = np.zeros((self.num_classes, self.num_tasks), dtype=np.int64)
        for c in range(self.num_classes):
            n_samples_table[c, 0] = self.num_samples_map_train[c]
        return n_samples_table
