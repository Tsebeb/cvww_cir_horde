import torch
from copy import deepcopy
from argparse import ArgumentParser

from utils import get_transform_from_dataloader, _get_unique_targets
from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """Class implementing the Learning Without Forgetting (LwF) approach
    described in https://arxiv.org/abs/1606.09282
    """

    # Weight decay of 0.0005 is used in the original article (page 4).
    # Page 4: "The warm-up step greatly enhances fine-tuning’s old-task performance, but is not so crucial to either our
    #  method or the compared Less Forgetting Learning (see Table 2(b))."
    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, fix_bn=False, eval_on_train=False, logger=None, exemplars_dataset=None, lamb=1, T=2,
                 use_early_stopping=True):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   fix_bn, eval_on_train, logger, exemplars_dataset, use_early_stopping)
        self.model_old = None
        self.lamb = lamb
        self.T = T

        # keep track of classes within current task
        self._classes_this_exp_list = None
        self._classes_this_exp = None

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Page 5: "lambda is a loss balance weight, set to 1 for most our experiments. Making lambda larger will favor
        # the old task performance over the new task’s, so we can obtain a old-task-new-task performance line by
        # changing lambda."
        parser.add_argument('--lamb', default=1, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        # Page 5: "We use T=2 according to a grid search on a held out set, which aligns with the authors’
        #  recommendations." -- Using a higher value for T produces a softer probability distribution over classes.
        parser.add_argument('--T', default=2, type=int, required=False,
                            help='Temperature scaling (default=%(default)s)')
        return parser.parse_known_args(args)

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

    def pre_train_process(self, t, trn_loader, val_loader):
        super().pre_train_process(t, trn_loader, val_loader)
        self._classes_this_exp = _get_unique_targets(trn_loader.dataset)

        if t > 0 and len(self.exemplars_dataset) > 0:
            exemplar_targets = _get_unique_targets(self.exemplars_dataset)
            self._classes_this_exp = torch.unique(self._classes_this_exp + exemplar_targets)
        self._classes_this_exp_list = self._classes_this_exp.tolist()
        self._classes_this_exp.to(self.device)

    def post_train_process(self, t, trn_loader, val_loader):
        """Runs after training all the epfochs of the task (after the train session)"""

        # Restore best and save model for future tasks
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()

        running_loss, running_ce_loss, running_lwf_loss = 0.0, 0.0, 0.0
        running_acc = 0
        running_elements = 0
        for images, targets in trn_loader:
            # Forward old model
            targets_old = None
            if t > 0:
                targets_old = self.model_old(images.to(self.device))
            # Forward current model
            outputs = self.model(images.to(self.device))
            train_targets = targets.clone().apply_(lambda x: self._classes_this_exp_list.index(x))
            loss, ce_loss, lwf_loss = self.train_criterion(outputs, train_targets.to(self.device), targets_old)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

            # Update statistics
            running_loss += loss.item() * targets.size(0)
            running_ce_loss += ce_loss.item() * targets.size(0)
            running_lwf_loss += lwf_loss.item() * targets.size(0)
            running_acc += torch.sum(torch.argmax(outputs, dim=1) == targets.to(self.device)).cpu().item()
            running_elements += targets.size(0)
        print(" | Train running: loss={:.3f}, ce_loss={:.3f}, lwf_loss={:.3f}, acc={:6.2f}% |".format(
            running_loss / running_elements, running_ce_loss / running_elements, running_lwf_loss / running_elements,
            running_acc / running_elements * 100), end="")

    def eval_early_stopping(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.inference_mode():
            total_loss, total_acc, total_num = 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward old model
                targets_old = None
                if self.model_old is not None:
                    targets_old = self.model_old(images.to(self.device))
                # Forward current model
                outputs = self.model(images.to(self.device))
                train_targets = targets.clone().apply_(lambda x: self._classes_this_exp_list.index(x))
                loss, _, _ = self.train_criterion(outputs, train_targets.to(self.device), targets_old)
                total_loss += loss.data.cpu().numpy().item() * len(targets)
                hits = self.calculate_metrics(outputs, targets)
                # Log
                total_acc += hits.sum().data.cpu().numpy().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc / total_num


    def eval(self, t, val_loader, calculate_loss=True):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc, total_num = 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward old model
                targets_old = None
                if self.model_old is not None:
                    targets_old = self.model_old(images.to(self.device))
                # Forward current model
                outputs = self.model(images.to(self.device))
                if calculate_loss:
                    loss = self.criterion(outputs, targets.to(self.device), targets_old)
                    total_loss += loss.data.cpu().numpy().item() * len(targets)
                hits = self.calculate_metrics(outputs, targets)
                # Log
                total_acc += hits.sum().data.cpu().numpy().item()
                total_num += len(targets)
        return total_loss / total_num if calculate_loss else 0.0, total_acc / total_num

    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Calculates cross-entropy with temperature scaling"""
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce

    def train_criterion(self, outputs, targets, outputs_old):
        lwf_loss, ce_loss = torch.tensor(0), torch.tensor(0)
        if self.model_old is not None:
            # Knowledge distillation loss for all previous tasks
            outputs_old_size = outputs_old.size(1)
            lwf_loss = self.lamb * self.cross_entropy(outputs[:, :outputs_old_size], outputs_old, exp=1.0 / self.T)
        ce_loss = torch.nn.functional.cross_entropy(outputs[:, self._classes_this_exp], targets)
        return ce_loss + lwf_loss, ce_loss, lwf_loss

    def criterion(self,  outputs, targets, outputs_old=None, return_seperate_loss=False):
        """Returns the loss value"""
        lwf_loss, ce_loss = torch.tensor(0), torch.tensor(0)
        if self.model_old is not None:
            # Knowledge distillation loss for all previous tasks
            outputs_old_size = outputs_old.size(1)
            lwf_loss = self.lamb * self.cross_entropy(outputs[:, :outputs_old_size], outputs_old, exp=1.0 / self.T)
        ce_loss = torch.nn.functional.cross_entropy(outputs, targets)
        if return_seperate_loss:
            return ce_loss + lwf_loss, ce_loss, lwf_loss
        return ce_loss + lwf_loss
