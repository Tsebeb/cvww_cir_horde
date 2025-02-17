import dill
import numpy as np
from argparse import ArgumentParser

from scenarios.manage_loaders import Manage_Loaders


#TODO: rename ugly naming convention
class Scenario_Loader(Manage_Loaders):
    """ Base class for random repetition """
    def __init__(self, datasets, validation, num_tasks, batch_size, num_workers, pin_memory, challenge_config):
        self.challenge_config = challenge_config
        # load config and setting exactly as used for the challenge
        with open(self.challenge_config, "rb") as f:
            self.config = dill.load(f)

        # ensure that there are enough experiences/tasks for the current runtime
        assert self.config["nexp"] <= num_tasks
        super(Scenario_Loader, self).__init__(datasets, validation, num_tasks, batch_size, num_workers, pin_memory)

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--challenge-config', type=str, required=True,
                            help='The base ')
        return parser.parse_known_args(args)

    def _generate_tasks(self):
        return self.config["n_samples_table"]
