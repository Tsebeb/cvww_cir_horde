import argparse
import numpy as np
from argparse import ArgumentParser

from scenarios.manage_loaders import Manage_Loaders


class Scenario_Loader(Manage_Loaders):
    """ This class implements the scenario as it is proposed in CIR for the CLVISION Challenge 2023 @ CVPR
         - this means each class has a initial geometric discovery probability
         - the reoccurence of the classes are fixed

    """
    def __init__(self, datasets, validation, num_tasks, batch_size, num_workers, pin_memory, imgs_per_task,
                 num_start_classes, num_start_samples, reoccurrence_prob_p, reoccurrence_prob_q):
        self.num_start_classes = num_start_classes
        self.num_start_samples = num_start_samples
        self.imgs_per_task = imgs_per_task
        self.reoccurrence_prob_p = reoccurrence_prob_p
        self.reoccurrence_prob_q = reoccurrence_prob_q
        assert self.imgs_per_task > 0
        super(Scenario_Loader, self).__init__(datasets, validation, num_tasks, batch_size, num_workers, pin_memory)

    @staticmethod
    def extra_parser(args):
        parser = argparse.ArgumentParser()
        parser.add_argument("--num-start-classes", type=int, default=None, required=False)
        parser.add_argument("--num-start-samples", type=int, default=None, required=False)
        parser.add_argument("--imgs-per-task", type=int, default=2000)
        parser.add_argument("--reoccurrence-prob-p", type=float, default=3.5)
        parser.add_argument("--reoccurrence-prob-q", type=float, default=20.0)
        return parser.parse_known_args(args)

    def _generate_tasks(self):
        rand_table = np.random.uniform(size=(self.num_classes, self.num_tasks))
        class_reoccurrence_prop = np.random.beta(a=self.reoccurrence_prob_p,
                                                 b=self.reoccurrence_prob_q,
                                                 size=self.num_classes)
        scenario_table = np.zeros_like(rand_table, dtype=int)
        for i, c_prob in enumerate(class_reoccurrence_prop):
            scenario_table[i, :] = rand_table[i, :] < c_prob
        number_classes_in_experience = np.sum(scenario_table, axis=0)
        min = np.min(number_classes_in_experience)
        # Repeat instances for classes with 0 classes
        while min == 0:
            new_table = np.random.uniform(size=(self.num_classes, self.num_tasks))
            rand_table[:, min == number_classes_in_experience] = new_table[:, min == number_classes_in_experience]
            scenario_table = np.zeros_like(rand_table)
            for i, c_prob in enumerate(class_reoccurrence_prop):
                scenario_table[i, :] = rand_table[i, :] < c_prob
            number_classes_in_experience = np.sum(scenario_table, axis=0)
            min = np.min(number_classes_in_experience)

        n_samples_table = scenario_table.copy().astype(np.int64)
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

        # overwrite initial start if properties are set
        if self.num_start_classes is not None:
            assert self.num_start_samples is not None, ("if manual overwrite of start classes is provided, a"
                                                        " equivalent number of start samples is needed")
            assert 0 < self.num_start_classes <= self.num_classes, ("Number of start classes need to be positive and less "
                                                                    "or equal than total number of classes")
            total_classes = list(range(self.num_classes))
            selected_classes = np.random.choice(total_classes, size=self.num_start_classes, replace=False)
            n_samples_table[:, 0] = 0
            n_samples_table[selected_classes, 0] = self.num_start_samples

        return n_samples_table


