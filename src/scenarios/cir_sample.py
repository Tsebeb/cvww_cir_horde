import argparse

import numpy as np
from argparse import ArgumentParser

from scenarios.manage_loaders import Manage_Loaders


class Scenario_Loader(Manage_Loaders):
    """ This class implements the scenario as it is proposed in CIR for the CLVISION Challenge 2023 @ CVPR
         - this means each class has a initial geometric discovery probability
         - the reoccurence of the classes are fixed

    """

    def __init__(self, datasets, validation, num_tasks, batch_size, num_workers, pin_memory, geometric_discovery_p, imgs_per_task,
                 class_reoccurrence):
        self.geometric_p = geometric_discovery_p
        self.imgs_per_task = imgs_per_task
        self.class_reoccurrence = class_reoccurrence

        assert 0 <= self.class_reoccurrence <= 1
        assert 0 <= self.geometric_p <= 1
        assert self.imgs_per_task > 0
        super(Scenario_Loader, self).__init__(datasets, validation, num_tasks, batch_size, num_workers, pin_memory)

    @staticmethod
    def extra_parser(args):
        parser = argparse.ArgumentParser()
        parser.add_argument("--geometric-discovery-p", type=float, default=0.5)
        parser.add_argument("--imgs-per-task", type=int, default=2000)
        parser.add_argument("--class-reoccurrence", type=float, default=0.2)

        return parser.parse_known_args(args)

    def _generate_tasks(self):
        # the size in the form of (m, n, k) where k is the power of the geometric)
        geometric_tab = np.random.geometric(self.geometric_p, (self.num_classes, 1, self.num_tasks))
        # How many are successful on first run these are present other are not
        geometric_tab[geometric_tab != 1] = 0
        geometric_tab = np.squeeze(geometric_tab)

        first_occ = np.zeros((self.num_classes,), dtype=np.int64)
        scenario_table = np.zeros((self.num_classes, self.num_tasks), dtype=np.int64)
        for i in range(self.num_classes):
            for j in range(self.num_tasks):
                if geometric_tab[i][j] == 1:
                    first_occ[i] = j

                    # First sample found sample the rest of the experiences with the repetition probability
                    scenario_table[i, j] = 1
                    nr_samples = self.num_tasks - 1 - j
                    if nr_samples > 0:
                        samples = np.random.random_sample(nr_samples)
                        for i_s, s in enumerate(samples):
                            if s < self.class_reoccurrence:
                                scenario_table[  i, j + i_s + 1] = 1
                                # if there are not enough samples than these will be cut off from the dataset
                    break

        n_samples_table = scenario_table.copy()
        number_classes_in_experience = np.sum(scenario_table, axis=0)
        for i in range(self.num_tasks):
            base_factor = self.imgs_per_task // number_classes_in_experience[i]
            n_samples_table[:, i] *= base_factor

            # Distribute rest randomly
            rest = self.imgs_per_task % number_classes_in_experience[i]
            if rest > 0:
                indices_present = np.squeeze(
                    np.argwhere(n_samples_table[:, i]))  # get indices greater 0 / Classes present
                chosen_idx = np.random.choice(indices_present, size=rest, replace=False)
                for c_idx in chosen_idx:
                    n_samples_table[c_idx, i] += 1
            assert np.sum(n_samples_table[:, i]) == self.imgs_per_task, "Something with distributing the classes went wrong"

        return n_samples_table


