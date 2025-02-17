import time

import numpy as np
import torch
from argparse import ArgumentParser

from torch.optim import Optimizer

from loggers.exp_logger import ExperimentLogger
from datasets.exemplars_dataset import ExemplarsDataset
from networks.cil_network import CIL_Net


def _set_learning_rate(optimizer: Optimizer, learning_rate):
    for g in optimizer.param_groups:
        g['lr'] = learning_rate

class Inc_Learning_Appr:
    """ Basic class for implementing incremental learning approaches WITH REPETITION
    Changes from FACIL include:
        - removal of warm-up and multi-softmax options
        - removal of task-aware acc --> only one accuracy since there is only one head
        - reduction of setting to single head -- future: grow the head as new classes are seen
    """

    def __init__(self, model, device, nepochs=[100], lr=[0.05], lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, fix_bn=False, eval_on_train=False, logger: ExperimentLogger = None,
                 exemplars_dataset: ExemplarsDataset = None, use_early_stopping=False):
        self.model: CIL_Net = model
        self.device = device
        self.nepochs = nepochs
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.momentum = momentum
        self.wd = wd
        self.logger = logger
        self.exemplars_dataset = exemplars_dataset
        self.fix_bn = fix_bn
        self.eval_on_train = eval_on_train
        self.use_early_stopping = use_early_stopping
        self.optimizer = None

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        return parser.parse_known_args(args)

    @staticmethod
    def exemplars_dataset_class():
        """Returns a exemplar dataset to use during the training if the approach needs it
        :return: ExemplarDataset class or None
        """
        return None

    @staticmethod
    def get_model_class():
        """
        Returns the model
        """
        return CIL_Net

    def _get_optimizer(self):
        """Returns the optimizer"""
        # NOTE: lr will be overwritten in the training loop!
        return torch.optim.Adam(self.model.parameters(), lr=0.1, weight_decay=self.wd)

    def train(self, t, trn_loader, val_loader):
        """Main train structure"""
        self.pre_train_process(t, trn_loader, val_loader)
        self.train_loop(t, trn_loader, val_loader)
        self.post_train_process(t, trn_loader, val_loader)

    def pre_train_process(self, t, trn_loader, val_loader):
        """Runs before training all epochs of the task (before the train session)"""
        pass

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        lr = self.lr[0]
        best_loss = np.inf
        patience = self.lr_patience
        best_model = self.model.get_copy()
        total_epochs = 0

        self.optimizer = self._get_optimizer()
        # Loop epochs
        for cur_epochs, cur_lr in zip(self.nepochs, self.lr):
            print("Setting LR to {:12.10f}".format(cur_lr))
            _set_learning_rate(self.optimizer, cur_lr)
            for e in range(cur_epochs):
                total_epochs += 1
                print('| Epoch {:3d}'.format(total_epochs), end="")
                clock0 = time.time()
                self.train_epoch(t, trn_loader)
                clock1 = time.time()
                # if self.eval_on_train:
                #     train_loss, train_acc = self.eval(t, trn_loader)
                #     clock2 = time.time()
                #     print('time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, acc={:5.1f}% |'.format(e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
                #     self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=train_loss, group="train")
                #     self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * train_acc, group="train")
                # else:
                print('time={:5.1f}s |'.format(clock1 - clock0), end='')

                if len(val_loader) > 0:
                    # Valid
                    clock3 = time.time()
                    valid_loss, valid_acc = self.eval_early_stopping(t, val_loader)
                    clock4 = time.time()
                    print(' Valid: time={:5.1f}s loss={:.3f}, acc={:5.1f}% |'.format(clock4 - clock3, valid_loss, 100 * valid_acc), end='')
                    self.logger.log_scalar(task=t, iter=total_epochs, name="loss", value=valid_loss, group="valid")
                    self.logger.log_scalar(task=t, iter=total_epochs, name="acc", value=100 * valid_acc, group="valid")

                    # Adapt learning rate - patience scheme - early stopping regularization
                    if self.use_early_stopping:
                        if valid_loss < best_loss:
                            # if the loss goes down, keep it as the best model and end line with a star ( * )
                            best_loss = valid_loss
                            best_model = self.model.get_copy()
                            patience = self.lr_patience
                            print(' *', end='')
                        else:
                            # if the loss does not go down, decrease patience
                            patience -= 1
                            if patience <= 0:
                                # if it runs out of patience, reduce the learning rate
                                lr /= self.lr_factor
                                print(' lr={:.1e}'.format(lr), end='')
                                if lr < self.lr_min:
                                    # if the lr decreases below minimum, stop the training session
                                    self.model.load_state_dict(best_model)
                                    print()
                                    return
                                # reset patience and recover best model so far to continue training
                                patience = self.lr_patience
                                _set_learning_rate(self.optimizer, lr)
                                self.model.load_state_dict(best_model)
                self.logger.log_scalar(task=t, iter=total_epochs, name="patience", value=patience, group="train")
                self.logger.log_scalar(task=t, iter=total_epochs, name="lr", value=lr, group="train")
                print()

    def post_train_process(self, t, trn_loader, val_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        pass

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()

        running_loss = 0.0
        running_acc = 0
        running_elements = 0
        for images, targets in trn_loader:
            # Forward current model
            outputs = self.model(images.to(self.device))
            loss = self.criterion(outputs, targets.to(self.device))
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

            # Update statistics
            running_loss += loss.item() * targets.size(0)
            running_acc += torch.sum(torch.argmax(outputs, dim=1) == targets.to(self.device)).cpu().item()
            running_elements += targets.size(0)
        if running_elements > 0:
            print(" | Train running: loss={:.3f}, acc={:6.2f}% |".format(running_loss / running_elements, running_acc / running_elements * 100), end="")

    def eval_early_stopping(self, t, val_loader):
        return self.eval(t, val_loader)

    def eval(self, t, val_loader, calculate_loss=True):
        """Contains the evaluation code"""
        with torch.inference_mode():
            total_loss, total_acc, total_num = 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward current model
                outputs = self.model(images.to(self.device))
                if calculate_loss:
                    loss = self.criterion(outputs, targets.to(self.device))
                    total_loss += loss.item() * len(targets)
                hits = self.calculate_metrics(outputs, targets)
                # Log
                total_acc += hits.sum().item()
                total_num += len(targets)
        return total_loss / total_num if calculate_loss else None, total_acc / total_num

    def calculate_metrics(self, outputs, targets):
        """Contains the main metrics"""
        # Task-Agnostic since there is only one head
        pred = outputs.argmax(1)
        hits = (pred == targets.to(self.device)).float()
        return hits

    def criterion(self, outputs, targets):
        """Returns the loss value"""
        # In this setting we only have one head for now - completely tag -- future: grow head as new classes appear
        return torch.nn.functional.cross_entropy(outputs, targets)
