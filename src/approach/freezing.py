import copy
from argparse import ArgumentParser

import torch

from utils import get_transform_from_dataloader, _get_unique_targets
from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """ Class implementing the finetuning baseline
    Changes from FACIL include:
        - removal of all-outputs argument
    """

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, fix_bn=False, eval_on_train=False, logger=None, exemplars_dataset=None,
                 use_early_stopping=False, task_ce_only=False, freeze_after_task=0):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   fix_bn, eval_on_train, logger, exemplars_dataset, use_early_stopping)
        self.freeze_after_task = freeze_after_task
        self.task_ce_only = task_ce_only
        self._classes_this_exp = None
        self._classes_this_exp_list = None
        self.__head_copy = None

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument("--task-ce-only", action="store_true")
        parser.add_argument("--freeze-after-task", default=0, help="freeze after task")
        return parser.parse_known_args(args)

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    def pre_train_process(self, t, trn_loader, val_loader):
        super().pre_train_process(t, trn_loader, val_loader)
        if self.task_ce_only:
            self._classes_this_exp = _get_unique_targets(trn_loader.dataset)
            if t > 0 and len(self.exemplars_dataset) > 0:
                exemplar_targets = _get_unique_targets(self.exemplars_dataset)
                self._classes_this_exp = torch.unique(self._classes_this_exp + exemplar_targets)
            self._classes_this_exp_list = self._classes_this_exp.tolist()
            self._classes_this_exp.to(self.device)
            self.__head_copy = copy.deepcopy(self.model.head)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, get_transform_from_dataloader(val_loader))

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
            if self.task_ce_only:
                outputs_train = outputs[:, self._classes_this_exp]
                targets_train = targets.clone().apply_(lambda x: self._classes_this_exp_list.index(x)).to(self.device)
                loss = self.criterion(outputs_train, targets_train)
            else:
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
            print(" | Train running: loss={:.3f}, acc={:6.2f}% |".format(running_loss / running_elements,
                                                                         running_acc / running_elements * 100), end="")
        if self.task_ce_only:
            # reset weights of head classes that are not in the current experience could be changed through weight decay or other
            # optimizer updates !
            with torch.no_grad():
                total_weights_copy = set(range(self.__head_copy.out_features))
                total_weights_copy = total_weights_copy.difference(set(self._classes_this_exp_list))
                for w in total_weights_copy:
                    self.model.head.weight[w, :] = self.__head_copy.weight[w, :]
                    self.model.head.bias[w] = self.__head_copy.bias[w]

    def eval_early_stopping(self, t, val_loader):
        with torch.inference_mode():
            total_loss, total_acc, total_num = 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward current model
                outputs = self.model(images.to(self.device))
                if self.task_ce_only:
                    outputs_train = outputs[:, self._classes_this_exp]
                    targets_train = targets.clone().apply_(lambda x: self._classes_this_exp_list.index(x)).to(self.device)
                    loss = self.criterion(outputs_train, targets_train)
                else:
                    loss = self.criterion(outputs, targets.to(self.device))
                total_loss += loss.item() * len(targets)
                hits = self.calculate_metrics(outputs, targets)
                # Log
                total_acc += hits.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc / total_num

    def post_train_process(self, t, trn_loader, val_loader):
        super().post_train_process(t, trn_loader, val_loader)
        if t >= self.freeze_after_task:
            self.model.freeze_backbone()
