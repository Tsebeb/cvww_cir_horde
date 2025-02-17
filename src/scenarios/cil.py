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

    def __init__(self, datasets, validation, num_tasks, batch_size, num_workers, pin_memory, num_start_classes, random_order):
        super(Scenario_Loader, self).__init__(datasets, validation, num_tasks, batch_size, num_workers, pin_memory)
        rest_classes = (self.num_classes - num_start_classes) % (num_tasks - 1)
        self.start_classes = num_start_classes
        self.random_order = random_order
        assert rest_classes == 0
        assert self.num_classes >= num_start_classes
        assert num_start_classes > 0

        if num_tasks > 1:
            self.classes_per_task = (self.num_classes - num_start_classes) // (num_tasks - 1)


    @staticmethod
    def extra_parser(args):
        parser = argparse.ArgumentParser()
        parser.add_argument("--num-start-classes", type=int, required=True)
        parser.add_argument("--random-order", action="store_true", default=False)
        return parser.parse_known_args(args)

    def _generate_tasks(self):
        n_samples_table = np.zeros((self.num_classes, self.num_tasks), dtype=np.int64)
        total_classes = list(range(self.num_classes))
        if self.random_order:
            for i in range(self.num_tasks):
                task_classes = random.sample(total_classes, k=self.start_classes if i == 0 else self.classes_per_task)
                for c in task_classes:
                    n_samples_table[c, i] = self.num_samples_map_train[c]
                _ = [total_classes.remove(f) for f in task_classes]
        else:
            for i in range(self.num_tasks):
                start_class = (i - 1) * self.classes_per_task + self.start_classes if i != 0 else 0
                end_class = (i) * self.classes_per_task + self.start_classes if i != 0 else self.start_classes
                for c in range(start_class, end_class):
                    n_samples_table[c, i] = self.num_samples_map_train[c]
        return n_samples_table
