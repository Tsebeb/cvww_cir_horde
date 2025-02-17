import time
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
import torch
from argparse import ArgumentParser

import torch.nn.functional as F
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset
from torchvision.transforms import Pad, CenterCrop, ToTensor, Normalize, RandomResizedCrop, RandomHorizontalFlip, \
    AugMix, Compose, Resize

from networks import resnet18_imagenet_lucir
from networks.fetril_network import FetrilNet
from networks.resnet18_lucir import resnet18_lucir, slimresnet18_lucir
from utils import override_transform_dataloader, get_transform_from_dataloader
from .incremental_learning import Inc_Learning_Appr


def _set_learning_rate(optimizer: Optimizer, learning_rate):
    for g in optimizer.param_groups:
        g['lr'] = learning_rate


_available_transforms = ["cifar_imagenet_augmix", "cifar_imagenet_test", "cifar_augmix", "cifar_test", "train"]

def get_adapted_transform(trn_loader: DataLoader, transform_type):
    if transform_type == "train":
        return get_transform_from_dataloader(trn_loader)
    elif transform_type == "cifar_test":
        return Compose([Pad(padding=4, fill=0),
                        CenterCrop(size=(32, 32)),
                        ToTensor(),
                        Normalize(mean=(0.5071, 0.4866, 0.4409), std=(0.2009, 0.1984, 0.2023))])
    elif transform_type == "cifar_augmix":
        return Compose([Pad(padding=4, fill=0),
                        RandomResizedCrop(size=(32, 32)),
                        RandomHorizontalFlip(p=0.5),
                        AugMix(severity=5, chain_depth=7),
                        ToTensor(),
                        Normalize(mean=(0.5071, 0.4866, 0.4409), std=(0.2009, 0.1984, 0.2023))])
    elif transform_type == "cifar_imagenet_test":
        return Compose([Resize(256),
                        CenterCrop(224),
                        ToTensor(),
                        Normalize(mean=(0.5071, 0.4866, 0.4409), std=(0.2009, 0.1984, 0.2023))])
    elif transform_type == "cifar_imagenet_augmix":
        return Compose([RandomResizedCrop(224),
                        AugMix(severity=5, chain_depth=7),
                        ToTensor(),
                        Normalize(mean=(0.5071, 0.4866, 0.4409), std=(0.2009, 0.1984, 0.2023))])
    else:
        raise NotImplementedError("Type not supported define here the datatransforms you need for pretraining")


class Lookahead(Optimizer):
    r"""PyTorch implementation of the lookahead wrapper.
    Lookahead Optimizer: https://arxiv.org/abs/1907.08610
    """

    def __init__(self, optimizer, la_steps=5, la_alpha=0.8, pullback_momentum="none"):
        """optimizer: inner optimizer
        la_steps (int): number of lookahead steps
        la_alpha (float): linear interpolation factor. 1.0 recovers the inner optimizer.
        pullback_momentum (str): change to inner optimizer momentum on interpolation update
        """
        self.optimizer = optimizer
        self._la_step = 0  # counter for inner optimizer
        self.la_alpha = la_alpha
        self._total_la_steps = la_steps
        pullback_momentum = pullback_momentum.lower()
        assert pullback_momentum in ["reset", "pullback", "none"]
        self.pullback_momentum = pullback_momentum

        self.state = defaultdict(dict)

        # Cache the current optimizer parameters
        for group in optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'] = torch.zeros_like(p.data)
                param_state['cached_params'].copy_(p.data)
                if self.pullback_momentum == "pullback":
                    param_state['cached_mom'] = torch.zeros_like(p.data)

    def __getstate__(self):
        return {
            'state': self.state,
            'optimizer': self.optimizer,
            'la_alpha': self.la_alpha,
            '_la_step': self._la_step,
            '_total_la_steps': self._total_la_steps,
            'pullback_momentum': self.pullback_momentum
        }

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_la_step(self):
        return self._la_step

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def _backup_and_load_cache(self):
        """Useful for performing evaluation on the slow weights (which typically generalize better)
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_params'])

    def _clear_and_load_backup(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])
                del param_state['backup_params']

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self, closure=None):
        """Performs a single Lookahead optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = self.optimizer.step(closure)
        self._la_step += 1

        if self._la_step >= self._total_la_steps:
            self._la_step = 0
            # Lookahead and cache the current optimizer parameters
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    p.data.mul_(self.la_alpha).add_(param_state['cached_params'], alpha=1.0 - self.la_alpha)  # crucial line
                    param_state['cached_params'].copy_(p.data)
                    if self.pullback_momentum == "pullback":
                        internal_momentum = self.optimizer.state[p]["momentum_buffer"]
                        self.optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(self.la_alpha).add_(
                            1.0 - self.la_alpha, param_state["cached_mom"])
                        param_state["cached_mom"] = self.optimizer.state[p]["momentum_buffer"]
                    elif self.pullback_momentum == "reset":
                        self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)
        return loss


class Appr(Inc_Learning_Appr):
    """Class implementing the finetuning baseline"""

    def __init__(self, model: FetrilNet, device, class_sim_method, fe_lr, fe_epochs, k_nearest_classes,
                 lucir_pretrain_model, lucir_fe_lr, lucir_fe_epochs, lucir_batch_size,
                 fe_data_augment, test_augment, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5,
                 clipgrad=10000, momentum=0, wd=0, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, use_early_stopping=False):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   fix_bn, eval_on_train, logger, exemplars_dataset, use_early_stopping)
        self.known_classes = []
        self.class_sim_method = class_sim_method
        self.classifier_type = self.model.classifier_type

        # Arguments for training the initial feature extractor f
        self.fe_lr = fe_lr
        self.fe_epochs = fe_epochs
        self.lucir_pretrain_model = lucir_pretrain_model
        self.lucir_batch_size = lucir_batch_size
        self.lucir_fe_lr = lucir_fe_lr
        self.lucir_fe_epochs = lucir_fe_epochs
        self.fe_data_augment = fe_data_augment

        # Classifier Feature Transforming arguments
        self.mean_embedding = torch.zeros((len(self.known_classes), self.model.out_size), device=self.device)
        # This is the number of pseudo features that represent each mean embedding (see Section 3.2 in the paper)
        self.s_features_per_class = torch.zeros((len(self.known_classes)), device=self.device)
        self.k_nearest = k_nearest_classes
        self.test_augment = test_augment
        if self.eval_on_train:
            print("Eval on Trian not supported on FetrIL. Disabling eval on train ... ")
            self.eval_on_train = False

        self._cur_val_to_idx_dict: Optional[dict] = None
        self._initial_training_done: bool = False
        assert len(fe_lr) == len(fe_epochs)
        assert len(lucir_fe_lr) == len(lucir_fe_epochs)
        assert k_nearest_classes > 0

    @staticmethod
    def get_model_class():
        return FetrilNet

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument("--k-nearest-classes", type=int, default=1)
        parser.add_argument("--class-sim-method", default="cosine", choices=["cosine", "random", "herding"])

        # Arguments for Feature Extractor Trainings
        parser.add_argument("--fe-lr", type=float, default=[0.01, 0.001, 0.0001], nargs="+")
        parser.add_argument("--fe-epochs", type=int, default=[80, 40, 170], nargs="+")
        parser.add_argument("--lucir-pretrain-model", type=str, default="resnet18_lucir", choices=["none", "resnet18_lucir", "slimresnet18_lucir", "resnet18_imagenet_lucir"])
        parser.add_argument("--lucir-fe-lr", type=float, default=[0.1, 0.01, 0.001], nargs="+")
        parser.add_argument("--lucir-fe-epochs", type=int, default=[30, 30, 30], nargs="+")
        parser.add_argument("--lucir-batch-size", type=int, default=128)
        parser.add_argument("--fe-data-augment", type=str, default="cifar_augmix", choices=_available_transforms)
        parser.add_argument("--test-augment", type=str, default="cifar_test", choices=_available_transforms)
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        params = self.model.head.parameters()
        return torch.optim.Adam(params, lr=0.01, weight_decay=self.wd)

    def pre_train_process(self, t, trn_loader, val_loader):
        # Count the number of unique classes present in exp
        self._cur_val_to_idx_dict = {}
        for i, (_, y) in enumerate(trn_loader.dataset):
            y = y.item()
            if y not in self._cur_val_to_idx_dict:
                self._cur_val_to_idx_dict[y] = [i]
            else:
                self._cur_val_to_idx_dict[y].append(i)

        # increase buffers in case of new classes!
        new_classes = []
        for class_idx in self._cur_val_to_idx_dict.keys():
            if class_idx not in self.known_classes:
                new_classes.append(class_idx)

        if len(new_classes) > 0:
            mean_embedding_new = torch.zeros((len(self.known_classes) + len(new_classes), self.model.out_size), device=self.device)
            mean_embedding_new[:len(self.known_classes), :] = self.mean_embedding[:, :]
            self.mean_embedding = mean_embedding_new

            s_features_per_class_new = torch.zeros((len(self.known_classes) + len(new_classes)), device=self.device)
            s_features_per_class_new[:len(self.known_classes)] = self.s_features_per_class[:]
            self.s_features_per_class = s_features_per_class_new
            self.known_classes += new_classes

        super(Appr, self).pre_train_process(t, trn_loader, val_loader)

        if not self._initial_training_done:
            self.train_feature_extractor(trn_loader, val_loader)

    def train_loop(self, t, trn_loader: DataLoader, val_loader):
        """Contains the epochs loop"""
        # train an initial feature extractor
        self._calculate_embedding_statistics(trn_loader)
        # Training of the head loop only the feature extractor is already frozen and no longer used!
        features = []
        targets = []
        with override_transform_dataloader(trn_loader, get_adapted_transform(trn_loader, self.test_augment), True) as test_train_loader:
            for image, y, in test_train_loader:
                fs = self.model.model(image.to(self.device))
                features.append(fs)
                targets.append(y.to(self.device))

        print("Transforming features into old classes")
        pseudo_features, pseudo_targets = self._transform_features(trn_loader)
        features += pseudo_features
        targets += pseudo_targets

        features = torch.vstack(features).cpu()
        targets = torch.cat(targets).cpu()

        if self.model.l2_norm:
            features = torch.nn.functional.normalize(features, p=2, dim=1)
        fetril_ds = TensorDataset(features, targets)
        # num_workers is 0 because elements are in memory and already on the device
        fetril_dl = DataLoader(dataset=fetril_ds, batch_size=trn_loader.batch_size, shuffle=True,
                               num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)

        if self.model.classifier_type == "fc":
            self.optimizer = self._get_optimizer()
            best_loss = np.inf
            best_model = None
            total_epochs = 0
            # Loop epochs
            for cur_epochs, current_lr in zip(self.nepochs, self.lr):
                patience = self.lr_patience
                _set_learning_rate(self.optimizer, current_lr)
                for e in range(cur_epochs):
                    total_epochs += 1
                    print('| Epoch {:3d}'.format(total_epochs), end="")
                    start_time = time.time()
                    self.train_epoch(t, fetril_dl)
                    end_time = time.time()
                    print('time={:5.1f}s |'.format(end_time - start_time), end='')

                    if len(val_loader) > 0:
                        # Valid
                        clock3 = time.time()
                        valid_loss, valid_acc = self.eval_early_stopping(t, val_loader)
                        clock4 = time.time()
                        print(' Valid: time={:5.1f}s loss={:.3f}, acc={:5.1f}% |'.format(clock4 - clock3, valid_loss, 100 * valid_acc), end='')
                        self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
                        self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")

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
                                    current_lr /= self.lr_factor
                                    print(' lr={:.1e}'.format(current_lr), end='')
                                    if current_lr < self.lr_min:
                                        # if the lr decreases below minimum, stop the training session
                                        self.model.load_state_dict(best_model)
                                        print()
                                        return
                                    # reset patience and recover best model so far to continue training
                                    patience = self.lr_patience
                                    _set_learning_rate(self.optimizer, current_lr)
                                    self.model.load_state_dict(best_model)
                    self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="train")
                    self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=current_lr, group="train")
                    print()

        elif self.model.classifier_type == "linear_svc":
            print("Fitting Linear SVC ... ")
            self.model.replace_svc_head(features, targets)
        else:
            raise NotImplementedError("Unsupported Classifier type")

    def post_train_process(self, t, trn_loader, val_loader):
        super().post_train_process(t, trn_loader, val_loader)
        self._initial_training_done = True
        for class_idx in self._cur_val_to_idx_dict.keys():
            # These are the s features that represent a class
            self.s_features_per_class[class_idx] = len(self._cur_val_to_idx_dict[class_idx])

    def _transform_features(self, trn_loader: DataLoader) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        pseudo_features = []
        pseudo_targets = []

        if self.class_sim_method == "random":
            # Randomly choose transfer features from all new classes
            for class_idx, s_num in enumerate(self.s_features_per_class):
                if s_num > 0:
                    single_batch_loader = DataLoader(trn_loader.dataset, batch_size=1, shuffle=True, num_workers=0)
                    for i, (image, y) in enumerate(single_batch_loader):
                        orig_features = self.model.model(image)
                        simulated_feats = orig_features - self.mean_embedding[y] + self.mean_embedding[class_idx]
                        pseudo_features.append(simulated_feats)
                        pseudo_targets.append(class_idx)

                        if i >= s_num:
                            break

        elif self.class_sim_method == "cosine":
            # calculate the n closest classes
            for class_idx, s_num in enumerate(self.s_features_per_class):
                if s_num > 0:
                    similarity = torch.cosine_similarity(self.mean_embedding[class_idx], self.mean_embedding)
                    # increase weight of the new classes so that only those can be chosen
                    for k in self._cur_val_to_idx_dict:
                        similarity[k] += 1
                    k_th_similar_cls = min(len(self._cur_val_to_idx_dict.keys()), self.k_nearest)
                    result = torch.topk(similarity, k=k_th_similar_cls)
                    total_indices = []
                    for index in result.indices:
                        total_indices += self._cur_val_to_idx_dict[index.item()]

                    # Get only the subset of the new classes
                    single_batch_loader = DataLoader(trn_loader.dataset, batch_size=1, num_workers=0,
                                                     sampler=SubsetRandomSampler(total_indices))
                    for i, (image, y) in enumerate(single_batch_loader):
                        orig_features = self.model.model(image.to(self.device))
                        simulated_feats = orig_features - self.mean_embedding[y.item()] + self.mean_embedding[class_idx]
                        pseudo_features.append(simulated_feats)
                        pseudo_targets.append(class_idx)

                        if i >= s_num:
                            break
        elif self.class_sim_method == "herding":
            raise NotImplementedError("Not yet implemented")
        else:
            raise RuntimeError("Unsupported class simulation method")
        return pseudo_features, [torch.tensor(pseudo_targets).long().to(self.device)]

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        # in the case of the Linear SVC just fit and apply
        self.model.head.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()

        running_loss = 0.0
        running_acc = 0
        running_elements = 0
        for features, targets in trn_loader:
            # Forward current model
            outputs = self.model.head(features.to(self.device))
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
        print(" | Train running: loss={:.3f}, acc={:6.2f}% |".format(running_loss / running_elements, running_acc / running_elements * 100), end="")

    def train_feature_extractor(self, trn_loader: DataLoader, val_loader: DataLoader):
        """Trains a new feature Extractor """
        self.model.train()
        def _train_model_with_schedule(model: torch.nn.Module, t_loader, v_loader, optimizer, schedule, pre_text):
            total_epochs = 0
            best_model_state_dict = None
            best_loss = None

            # Training loop
            for schedule in schedule:
                patience = self.lr_patience
                current_lr = schedule[0]
                _set_learning_rate(optimizer, current_lr)
                for epoch in range(schedule[1]):
                    total_epochs += 1
                    metric_ce_loss = 0.0
                    hits, total_elements = 0, 0
                    model.train()

                    start_time = time.time()
                    for x, y in t_loader:
                        x_device = x.to(self.device)
                        y_device = y.to(self.device)
                        ce_out = model(x_device)
                        ce_loss = torch.nn.functional.cross_entropy(ce_out, y_device)
                        metric_ce_loss += ce_loss.detach().item() * y.size(0)

                        optimizer.zero_grad()
                        ce_loss.backward()
                        optimizer.step()
                        # ---
                        hits += torch.sum(ce_out.argmax(dim=1) == y_device).detach().item()
                        total_elements += y.size(0)
                    metric_ce_loss /= total_elements
                    end_time = time.time()
                    print(f"{pre_text} | Epoch {total_epochs} | Train running: loss={metric_ce_loss:.3f}, "
                          f"acc={hits / total_elements * 100.0:6.2f}% | time={end_time-start_time:5.1f}s |", end="")

                    # Evaluate Early Stopping
                    if len(v_loader) > 0:
                        start_time = time.time()
                        with torch.inference_mode():
                            model.eval()
                            val_loss, acc = 0.0, 0.0
                            val_num_elements, val_hits = 0, 0
                            for x, y in v_loader:
                                y_device = y.to(self.device)
                                val_pred = model(x.to(self.device))
                                loss = F.cross_entropy(val_pred, y_device)
                                val_hits += torch.sum(val_pred.argmax(dim=1) == y_device).detach().item()
                                val_loss += loss.item() * y.size(0)
                                val_num_elements += y.size(0)
                        end_time = time.time()
                        val_loss /= val_num_elements
                        print(f" Valid: time={end_time - start_time:5.1f}s loss={val_loss:.3f}, acc={val_hits / val_num_elements * 100.0:5.1f}% |", end="")
                        self.logger.log_scalar(task=0, iter=total_epochs, name=f"{pre_text}_fe_loss", value=val_loss, group="valid")
                        self.logger.log_scalar(task=0, iter=total_epochs, name=f"{pre_text}_fe_acc", value=val_hits / val_num_elements * 100.0, group="valid")

                        if self.use_early_stopping:
                            if best_loss is None or val_loss < best_loss:
                                # if the loss goes down, keep it as the best model and end line with a star ( * )
                                best_loss = val_loss
                                best_model_state_dict = model.state_dict()
                                patience = self.lr_patience
                                print(' *', end='')
                            else:
                                # if the loss does not go down, decrease patience
                                patience -= 1
                                if patience <= 0:
                                    # if it runs out of patience, reduce the learning rate
                                    current_lr /= self.lr_factor
                                    print(' lr={:.1e}'.format(current_lr), end='')
                                    if current_lr < self.lr_min:
                                        model.load_state_dict(best_model_state_dict)
                                        print()
                                        break
                                    # reset patience and recover best model so far to continue training
                                    patience = self.lr_patience
                                    _set_learning_rate(optimizer, current_lr)
                                    model.load_state_dict(best_model_state_dict)
                    self.logger.log_scalar(task=0, iter=total_epochs, name=f"{pre_text}_fe_patience", value=patience, group="train")
                    self.logger.log_scalar(task=0, iter=total_epochs, name=f"{pre_text}_fe_lr", value=current_lr, group="train")
                    print()  # for Line break!
        if self.lucir_pretrain_model != "none":
            if self.lucir_pretrain_model == "resnet18_lucir":
                lucir_model = resnet18_lucir(num_classes=len(self.known_classes))
            elif self.lucir_pretrain_model == "slimresnet18_lucir":
                lucir_model = slimresnet18_lucir(num_classes=len(self.known_classes))
            elif self.lucir_pretrain_model == "resnet18_imagenet_lucir":
                lucir_model = resnet18_imagenet_lucir(num_classes=len(self.known_classes))
            else:
                raise RuntimeError("Unsupported LUCIR Pretrain Model")
            lucir_model.to(self.device)
            # LR will be overwritten
            optimizer = Adam(lucir_model.parameters(), 0.1, weight_decay=self.wd)
            _train_model_with_schedule(lucir_model, trn_loader, val_loader, optimizer,
                                       zip(self.lucir_fe_lr, self.lucir_fe_epochs), "LUCIR Pre Training")

            # Copy from lucir variant where it applies
            lucir_state_dict = lucir_model.state_dict()
            for key in list(lucir_state_dict.keys()):
                if key.startswith('fc'):
                    del lucir_state_dict[key]
            self.model.load_state_dict(lucir_state_dict, strict=False)

        # Copy Model Parameters to the original model - Temporary disable the l2 normalization for the forward
        normalize_l2_temp_disable = self.model.l2_norm
        self.model.l2_norm = False
        optimizer = Adam(self.model.parameters(), 0.1, weight_decay=self.wd)
        optimizer = Lookahead(optimizer)
        with override_transform_dataloader(trn_loader, get_adapted_transform(trn_loader, self.fe_data_augment), True) as test_train_loader:
            _train_model_with_schedule(self.model, test_train_loader, val_loader, optimizer,
                                       zip(self.fe_lr, self.fe_epochs), "FE Training")

        self.model.freeze_backbone()
        self.model.model.eval()
        self.model.l2_norm = normalize_l2_temp_disable

    def _calculate_embedding_statistics(self, trn_loader: DataLoader):
        with torch.inference_mode():
            with override_transform_dataloader(trn_loader, get_adapted_transform(trn_loader, self.test_augment), False) as test_train_loader:
                for cls_idx in self._cur_val_to_idx_dict:
                    cls_sampler = SubsetRandomSampler(indices=self._cur_val_to_idx_dict[cls_idx])
                    cls_loader = DataLoader(test_train_loader.dataset, batch_size=test_train_loader.batch_size, shuffle=False, sampler=cls_sampler)
                    all_feats = None
                    for image, target in cls_loader:
                        image = image.to(self.device)
                        features = self.model.model(image)
                        all_feats = features if all_feats is None else torch.vstack((all_feats, features))

                    # Calculate Mean
                    self.mean_embedding[cls_idx, :] = torch.mean(all_feats, dim=0)

    def calculate_metrics(self, outputs, targets):
        """Contains the main Task-Agnostic metrics"""
        pred = outputs.argmax(dim=1)
        hits = (pred == targets.to(self.device)).float()
        return hits


