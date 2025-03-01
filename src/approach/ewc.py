import copy
from typing import Optional

import torch
import itertools
from argparse import ArgumentParser

from datasets.exemplars_dataset import ExemplarsDataset
from utils import get_transform_from_dataloader, _get_unique_targets
from .incremental_learning import Inc_Learning_Appr


class Appr(Inc_Learning_Appr):
    """Class implementing the Elastic Weight Consolidation (EWC) approach
    described in http://arxiv.org/abs/1612.00796
    """

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, fix_bn=False, eval_on_train=False, logger=None, exemplars_dataset=None, lamb=5000,
                 alpha=0.5, fi_sampling_type='max_pred', fi_num_samples=-1, use_early_stopping=False, task_ce_only=True):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   fix_bn, eval_on_train, logger, exemplars_dataset, use_early_stopping)
        self.lamb = lamb
        self.alpha = alpha
        self.sampling_type = fi_sampling_type
        self.num_samples = fi_num_samples
        self.task_ce_only = task_ce_only

        # In all cases, we only keep importance weights for the model, but not for the heads.
        feat_ext = self.model.model
        # Store current parameters as the initial parameters before first task starts
        self.older_params = {n: p.clone().detach() for n, p in feat_ext.named_parameters() if p.requires_grad}
        # Store fisher information weight importance
        self.fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in feat_ext.named_parameters()
                       if p.requires_grad}
        self.first_iter_done = False
        self._classes_this_exp = None
        self._classes_this_exp_list = None
        self.__head_copy: Optional[torch.nn.Linear] = None

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Eq. 3: "lambda sets how important the old task is compared to the new one"
        parser.add_argument('--lamb', default=5000, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        # Define how old and new fisher is fused, by default it is a 50-50 fusion
        parser.add_argument('--alpha', default=0.5, type=float, required=False,
                            help='EWC alpha (default=%(default)s)')
        parser.add_argument('--fi-sampling-type', default='max_pred', type=str, required=False,
                            choices=['true', 'max_pred', 'multinomial'],
                            help='Sampling type for Fisher information (default=%(default)s)')
        parser.add_argument('--fi-num-samples', default=-1, type=int, required=False,
                            help='Number of samples for Fisher information (-1: all available) (default=%(default)s)')
        parser.add_argument("--task-ce-only", action="store_true")
        return parser.parse_known_args(args)

    def pre_train_process(self, t, trn_loader, val_loader):
        super().pre_train_process(t, trn_loader, val_loader)
        self._classes_this_exp = _get_unique_targets(trn_loader.dataset)

        if self.task_ce_only:
            if t > 0 and len(self.exemplars_dataset) > 0:
                exemplar_targets = _get_unique_targets(self.exemplars_dataset)
                self._classes_this_exp = torch.unique(self._classes_this_exp + exemplar_targets)
            self._classes_this_exp_list = self._classes_this_exp.tolist()
            self._classes_this_exp.to(self.device)
            self.__head_copy = copy.deepcopy(self.model.head)

    def compute_fisher_matrix_diag(self, trn_loader):
        # Store Fisher Information
        fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self.model.model.named_parameters()
                  if p.requires_grad}
        # Compute fisher information for specified number of samples -- rounded to the batch size
        n_samples_batches = (self.num_samples // trn_loader.batch_size + 1) if self.num_samples > 0 \
            else (len(trn_loader.dataset) // trn_loader.batch_size)
        # Do forward and backward pass to compute the fisher information
        self.model.train()
        for images, targets in itertools.islice(trn_loader, n_samples_batches):
            outputs = self.model.forward(images.to(self.device))

            if self.sampling_type == 'true':
                # Use the labels to compute the gradients based on the CE-loss with the ground truth
                preds = targets.to(self.device)
            elif self.sampling_type == 'max_pred':
                # Not use labels and compute the gradients related to the prediction the model has learned
                preds = outputs.argmax(1).flatten()
            elif self.sampling_type == 'multinomial':
                # Use a multinomial sampling to compute the gradients
                probs = torch.nn.functional.softmax(outputs, dim=1)
                preds = torch.multinomial(probs, len(targets)).flatten()

            loss = torch.nn.functional.cross_entropy(outputs, preds)
            self.optimizer.zero_grad()
            loss.backward()
            # Accumulate all gradients from loss with regularization
            for n, p in self.model.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2) * len(targets)
        # Apply mean across all samples
        n_samples = n_samples_batches * trn_loader.batch_size
        fisher = {n: (p / n_samples) for n, p in fisher.items()}
        return fisher

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

    def post_train_process(self, t, trn_loader, val_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        # Store current parameters for the next task
        self.older_params = {n: p.clone().detach() for n, p in self.model.model.named_parameters() if p.requires_grad}

        # calculate Fisher information
        curr_fisher = self.compute_fisher_matrix_diag(trn_loader)
        # merge fisher information, we do not want to keep fisher information for each task in memory
        for n in self.fisher.keys():
            # Added option to accumulate fisher over time with a pre-fixed growing alpha
            if self.alpha == -1:
                alpha = (sum(self.model.task_cls[:t]) / sum(self.model.task_cls)).to(self.device)
                self.fisher[n] = alpha * self.fisher[n] + (1 - alpha) * curr_fisher[n]
            else:
                self.fisher[n] = (self.alpha * self.fisher[n] + (1 - self.alpha) * curr_fisher[n])

        self.first_iter_done = True

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
            else:
                outputs_train = outputs
                targets_train = targets.to(self.device)
            loss = self.criterion(outputs_train, targets_train)

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

        # reset weights of head classes that are not in the current experience could be changed through weight decay or other
        # optimizer updates !
        if self.task_ce_only:
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
                else:
                    outputs_train = outputs
                    targets_train = targets.to(self.device)
                loss = self.criterion(outputs_train, targets_train)
                total_loss += loss.item() * len(targets)
                hits = self.calculate_metrics(outputs, targets)
                # Log
                total_acc += hits.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc / total_num

    def criterion(self, outputs, targets):
        """Returns the loss value"""
        loss = 0
        if self.first_iter_done:
            loss_reg = 0
            # Eq. 3: elastic weight consolidation quadratic penalty
            for n, p in self.model.model.named_parameters():
                if n in self.fisher.keys():
                    loss_reg += torch.sum(self.fisher[n] * (p - self.older_params[n]).pow(2)) / 2
            loss += self.lamb * loss_reg
        # Current cross-entropy loss -- with exemplars use all heads
        # if len(self.exemplars_dataset) > 0:
        #     return loss + torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return loss + torch.nn.functional.cross_entropy(outputs, targets)
