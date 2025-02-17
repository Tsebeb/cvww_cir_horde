import numpy as np
from argparse import ArgumentParser

from scenarios.manage_loaders import Manage_Loaders


class Scenario_Loader(Manage_Loaders):
    """ Base class for random repetition """

    def __init__(self, datasets, validation, num_tasks, batch_size, num_workers, pin_memory, class_presence,
                 low_thresh_normal, high_thresh_normal):
        self.class_presence = class_presence
        self.low_thresh_normal = low_thresh_normal
        self.high_thresh_normal = high_thresh_normal

        # Sanity checks
        assert 0 < class_presence < 1
        assert 0 <= low_thresh_normal <= 1
        assert 0 <= high_thresh_normal <= 1
        assert low_thresh_normal < high_thresh_normal
        super(Scenario_Loader, self).__init__(datasets, validation, num_tasks, batch_size, num_workers, pin_memory)

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--class-presence', default=0.5, type=float, required=False,
                            help='Probability of a class being in a task (default=%(default)s)')
        parser.add_argument("--low-thresh-normal", default=0.1, type=float)
        parser.add_argument("--high-thresh-normal", default=0.9, type=float)
        return parser.parse_known_args(args)

    def _generate_tasks(self):
        # Initialize class presence in the task
        cls_presence = np.random.uniform(size=(self.num_classes, self.num_tasks))
        cls_presence = (cls_presence > self.class_presence).astype(int)

        # Initialize percentage of samples per class per task
        cls_num_samples = np.random.normal(size=(self.num_classes, self.num_tasks), loc=0.5)
        cls_num_samples[cls_num_samples < self.low_thresh_normal] = self.low_thresh_normal
        cls_num_samples[cls_num_samples > self.high_thresh_normal] = self.high_thresh_normal

        # Put all together -- multiply the number of samples of a class by the percentage and multiply by presence
        view = np.tile(self.num_samples_map_train[:, None], (1, self.num_tasks))
        return ((cls_num_samples * view) * cls_presence).astype(int)
