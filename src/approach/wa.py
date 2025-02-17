import torch
from copy import deepcopy
from argparse import ArgumentParser

from torch.utils.data import DataLoader, ConcatDataset

from networks.wa_network import WeightAlignNet
from utils import _get_unique_targets, get_transform_from_dataloader
from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """Maintaining Discrimination and Fairness in Class Incremental Learning. CVPR2020
    described in https://arxiv.org/abs/1911.07053
    """

    # Weight decay of 0.0005 is used in the original article (page 4).
    # Page 4: "The warm-up step greatly enhances fine-tuning’s old-task performance, but is not so crucial to either our
    #  method or the compared Less Forgetting Learning (see Table 2(b))."
    def __init__(self, model, device, initial_nepochs, initial_lr, initial_wd, clip_head_weights, nepochs=100,
                 lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,momentum=0, wd=0, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, T=2, use_early_stopping=True):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   fix_bn, eval_on_train, logger, exemplars_dataset, use_early_stopping)
        self.model_old = None
        self.T = T
        assert len(initial_nepochs) == len(initial_lr)
        self.initial_nepochs = initial_nepochs
        self.initial_lr = initial_lr
        self.initial_wd = initial_wd

        # Save for incremental steps
        self._temp_lr = self.lr
        self._temp_nepochs = self.nepochs
        self._temp_wd = self.wd
        self.clip_head_weights = clip_head_weights

        self.known_classes = set()
        self.new_classes = set()
        self.classes_this_exp = set()

    @staticmethod
    def get_model_class():
        return WeightAlignNet

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument("--initial-nepochs", default=[100, 50, 50, 50], type=int, nargs="+", help="Training time for the first iteration")
        parser.add_argument("--initial-lr", default=[0.1, 0.01, 0.001, 0.0001], type=float, nargs="+", help="Training learning rate for the first iteration")
        parser.add_argument("--initial-wd", default=0.0005, type=float, help="Weight decay for the initial training of the model")
        parser.add_argument("--clip-head-weights", action="store_true", help="Whether the weights of the classifier head should be clipped after each update to positive values only!")
        parser.add_argument('--T', default=2, type=int, required=False, help='Temperature scaling (default=%(default)s) for the knowledge distillation')
        return parser.parse_known_args(args)

    def _get_total_classes(self):
        return len(self.new_classes) + len(self.known_classes)

    def pre_train_process(self, t, trn_loader, val_loader):
        super().pre_train_process(t, trn_loader, val_loader)
        # Calculate number of classes here and a mapping from class_idx to indices in the dataset
        self.classes_this_exp = _get_unique_targets(trn_loader.dataset).tolist()
        for class_idx in self.classes_this_exp:
            if class_idx not in self.known_classes:
                self.new_classes.add(class_idx)

        if t > 0:
            self.lr = self._temp_lr
            self.nepochs = self._temp_nepochs
            self.wd = self._temp_wd
        else:
            self.lr = self.initial_lr
            self.nepochs = self.initial_nepochs
            self.wd = self.initial_wd

        print("Classes This Exp", self.classes_this_exp)
        print("Known Classes", self.known_classes)
        print("New Classes", self.new_classes)
        print("Total Classes", self._get_total_classes())

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = DataLoader(ConcatDataset([trn_loader.dataset, self.exemplars_dataset]),
                                    batch_size=trn_loader.batch_size,
                                    shuffle=True,
                                    num_workers=trn_loader.num_workers,
                                    pin_memory=trn_loader.pin_memory)

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)
        if t > 0:
            self.model.weight_align(list(self.known_classes.difference(set(self.classes_this_exp))), self.classes_this_exp)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, get_transform_from_dataloader(val_loader))

    def post_train_process(self, t, trn_loader, val_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        self.known_classes.update(self.new_classes)
        self.new_classes = set()
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()

        running_loss = 0.0
        running_acc = 0
        running_elements = 0
        for images, targets in trn_loader:
            images, targets = images.to(self.device), targets.to(self.device)
            pred = self.model(images)
            # Forward current model
            loss = self._compute_loss(images, pred, targets)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
            if self.clip_head_weights:
                self.model.clip_head_weights_positive()

            # Update statistics
            running_loss += loss.item() * targets.size(0)
            running_acc += torch.sum(torch.argmax(pred, dim=1) == targets).cpu().item()
            running_elements += targets.size(0)
        print(" | Train running: loss={:.3f}, acc={:6.2f}% |".format(running_loss / running_elements, running_acc / running_elements * 100), end="")

    def eval_early_stopping(self, t, val_loader):
        with torch.inference_mode():
            self.model.eval()
            val_loss = 0.0
            val_hits = 0
            val_elements = 0
            for images, targets in val_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                pred = self.model(images)
                # Forward current model
                loss = self._compute_loss(images, pred, targets)

                # Update statistics
                val_loss += loss.item() * targets.size(0)
                val_hits += torch.sum(torch.argmax(pred, dim=1) == targets).cpu().item()
                val_elements += targets.size(0)
            return val_loss / val_elements, val_hits / val_elements

    def _compute_loss(self, images, pred, targets):
        loss_ce = torch.nn.functional.cross_entropy(pred, targets)
        if self.model_old is None:
            return loss_ce

        kd_lambda = (self._get_total_classes() - len(self.classes_this_exp)) / self._get_total_classes()
        old_model_pred = self.model_old(images)
        loss_kd = self._KD_loss(pred[:, :old_model_pred.size(1)],
                                old_model_pred,
                                self.T)
        return (1-kd_lambda) * loss_ce + kd_lambda * loss_kd

    def _KD_loss(self, pred, pred_old, T):
        pred = torch.log_softmax(pred / T, dim=1)
        pred_old = torch.softmax(pred_old / T, dim=1)
        return -1 * torch.mul(pred_old, pred).sum() / pred.shape[0]
