import time
from copy import deepcopy
import numpy as np
import torch
from argparse import ArgumentParser

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from networks.praka_network import PRAKA_Net
from utils import _get_unique_targets
from .incremental_learning import Inc_Learning_Appr
import torch.nn.functional as F


def _set_learning_rate(optimizer: Optimizer, learning_rate):
    for g in optimizer.param_groups:
        g['lr'] = learning_rate

class Appr(Inc_Learning_Appr):
    """Class implementing the PRAKA (prototype augmentation and self-supervision) approach published at the ICCV 2023
    described in https://openaccess.thecvf.com/content/ICCV2023/papers/Shi_Prototype_Reminiscence_and_Augmented_Asymmetric_Knowledge_Aggregation_for_Non-Exemplar_Class-Incremental_ICCV_2023_paper.pdf
    Original reference code is implemented here: https://github.com/ShiWuxuan/PRAKA/tree/master
    """

    def __init__(self, model: PRAKA_Net, device, nepochs, lr, lr_min, lr_factor, lr_patience, protoaug_weight, kd_weight,
                 temp, use_cosine_annealing_scheduler, cosine_annealing_rate, clipgrad=10000, momentum=0, wd=0,
                 fix_bn=False, eval_on_train=False, logger=None, exemplars_dataset=None, use_early_stopping=True):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   fix_bn, eval_on_train, logger, exemplars_dataset, use_early_stopping)
        # Parameters
        self.protoaug_weight = protoaug_weight
        self.kd_weight = kd_weight
        self.temp = temp
        self.use_cosine_annealing_scheduler = use_cosine_annealing_scheduler
        self.cosine_annealing_rate = cosine_annealing_rate

        # State variables
        self.prototype = None
        self.old_model = None

        # Class
        self.known_classes = []
        self.new_classes = []
        self.classes_this_exp = []

    @staticmethod
    def get_model_class():
        return PRAKA_Net

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument("--protoaug-weight", default=15.0, type=float, help="protoAug loss weight")
        parser.add_argument("--kd-weight", default=15.0, type=float, help="knowledge distillation loss weight")
        parser.add_argument("--use-cosine-annealing-scheduler", action="store_true", help="Use a cosine annealing learning rate scheduler")
        parser.add_argument("--cosine-annealing-rate", type=int, default=32)
        parser.add_argument("--temp", default=0.1, type=float, help="training temperature factor tau used to adjust the logits for cross entropy and knowledge distillation")
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr[0], weight_decay=self.wd)

    def _get_total_classes(self):
        return len(self.new_classes) + len(self.known_classes)

    def _get_classes_previous_learned(self):
        known_classes_set = set(self.known_classes)
        this_exp_classes = set(self.classes_this_exp)
        return list(known_classes_set.difference(this_exp_classes))

    def pre_train_process(self, t, trn_loader, val_loader):
        super().pre_train_process(t, trn_loader, val_loader)
        self.model.eval()

        # Calculate number of classes here and a mapping from class_idx to indices in the dataset
        self.classes_this_exp = _get_unique_targets(trn_loader.dataset).tolist()
        for class_idx in self.classes_this_exp:
            if class_idx not in self.known_classes:
                self.new_classes.append(class_idx)

        # ensure that heads gets adjusted / usually is covered by main
        self.model.modify_head(self._get_total_classes())
        self.model.train()
        self.model.to(self.device)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        best_loss = np.inf
        patience = self.lr_patience
        best_model = self.model.get_copy()
        total_epochs = 0

        self.optimizer = self._get_optimizer()
        if self.use_cosine_annealing_scheduler:
            scheduler = CosineAnnealingLR(self.optimizer, T_max=self.cosine_annealing_rate)
        # Loop epochs
        for cur_epochs, lr in zip(self.nepochs, self.lr):
            print("Setting LR to {:12.10f}".format(lr))
            _set_learning_rate(self.optimizer, lr)
            for e in range(cur_epochs):
                total_epochs += 1
                if self.use_cosine_annealing_scheduler:
                    scheduler.step()
                # Train
                print('| Epoch {:3d}'.format(total_epochs), end="")
                clock0 = time.time()
                self.train_epoch(t, trn_loader)
                clock1 = time.time()
                if self.eval_on_train:
                    train_loss, train_acc = self.eval(t, trn_loader)
                    clock2 = time.time()
                    print('time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, acc={:5.1f}% |'.format(clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
                    self.logger.log_scalar(task=t, iter=total_epochs, name="loss", value=train_loss, group="train")
                    self.logger.log_scalar(task=t, iter=total_epochs, name="acc", value=100 * train_acc, group="train")
                else:
                    print('time={:5.1f}s | Train: skip eval |'.format(clock1 - clock0), end='')

                if len(val_loader) > 0:
                    # Valid
                    clock3 = time.time()
                    valid_loss, valid_acc = self.eval_early_stopping(t, val_loader)
                    clock4 = time.time()
                    print(' Valid: time={:5.1f}s loss={:.3f}, acc={:5.1f}% |'.format(clock4 - clock3, valid_loss,
                                                                                     100 * valid_acc), end='')
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
                self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="train")
                print()

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()

        running_loss = 0.0
        running_acc = 0
        running_acc_joint = 0
        running_elements = 0
        for images, targets in trn_loader:
            orig_channels, orig_height, orig_width = images.size(1), images.size(2), images.size(3)
            # self-supervised learning based label augmentation
            images = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1)
            images = images.view(-1, orig_channels, orig_height, orig_width)
            joint_targets = torch.stack([targets * 4 + k for k in range(4)], 1).view(-1)
            joint_targets = joint_targets.to(self.device)
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs, joint_outputs, loss = self._compute_loss(images, joint_targets, targets)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

            # Update statistics
            running_loss += loss.item() * targets.size(0)
            running_acc += torch.sum(torch.argmax(outputs, dim=1) == targets).cpu().item()
            running_acc_joint += torch.sum(torch.argmax(joint_outputs, dim=1) == joint_targets).cpu().item()
            running_elements += targets.size(0)
        if running_elements > 0:
            print(" | Train running: loss={:.3f}, acc={:6.2f}%, augmented_acc={:6.2f}%|".format(running_loss / running_elements, running_acc / running_elements * 100, (running_acc_joint / 4.0) / running_elements * 100), end="")

    def eval_early_stopping(self, t, val_loader):
        with torch.inference_mode():
            self.model.eval()

            val_loss = 0.0
            val_hits, num_elements = 0, 0
            for images, targets in val_loader:
                orig_channels, orig_height, orig_width = images.size(1), images.size(2), images.size(3)
                # self-supervised learning based label augmentation
                images = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1)
                images = images.view(-1, orig_channels, orig_height, orig_width)
                joint_targets = torch.stack([targets * 4 + k for k in range(4)], 1).view(-1)
                joint_targets = joint_targets.to(self.device)
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs, joint_outputs, loss = self._compute_loss(images, joint_targets, targets)

                # Update statistics
                val_loss += loss.item() * targets.size(0)
                val_hits += torch.sum(torch.argmax(outputs, dim=1) == targets).cpu().item()
                num_elements += targets.size(0)

            return val_loss / num_elements, val_hits / num_elements

    def post_train_process(self, t, trn_loader, val_loader):
        self.protoSave(t, trn_loader)
        super().post_train_process(t, trn_loader, val_loader)
        # Reset old and new classes
        self.known_classes += self.new_classes
        self.new_classes = []

        self.old_model = deepcopy(self.model)
        self.old_model.freeze_all()
        self.old_model.to(self.device)
        self.old_model.eval()

    def protoSave(self, t, trn_loader: DataLoader):
        features = []
        labels = []
        self.model.eval()
        with torch.no_grad():
            for i, (images, target) in enumerate(trn_loader):
                feature = self.model.get_features(images.to(self.device))
                labels.append(target.numpy())
                features.append(feature.cpu().numpy())
        labels = np.concatenate(labels)
        features = np.vstack(features)

        prototype = {}
        class_label = []
        for item in self.classes_this_exp:
            index = np.where(item == labels)[0]
            class_label.append(item)
            feature_classwise = features[index]
            prototype[item] = np.mean(feature_classwise, axis=0)

        if t == 0:
            self.prototype = prototype
        else:
            self.prototype.update(prototype)

    def _compute_loss(self, imgs, joint_target, target):
        feature = self.model.get_features(imgs)
        joint_preds = self.model.orientation_class_head(feature)
        single_preds = self.model.head(feature[::4])
        joint_loss = F.cross_entropy(joint_preds / self.temp, joint_target)
        signle_loss = F.cross_entropy(single_preds / self.temp, target)

        agg_preds = 0
        for i in range(4):
            agg_preds = agg_preds + joint_preds[i::4, i::4] / 4

        distillation_loss = F.kl_div(F.log_softmax(single_preds, 1),
                                     F.softmax(agg_preds.detach(), 1),
                                     reduction='batchmean')

        if len(self._get_classes_previous_learned()) == 0:
            return single_preds, joint_preds, joint_loss + signle_loss + distillation_loss
        else:
            feature_old = self.old_model.get_features(imgs)
            loss_kd = torch.dist(feature, feature_old, 2)

            proto_aug = []
            proto_aug_label = []
            old_class_list = self._get_classes_previous_learned()
            for _ in range(feature.shape[0] // 4):  # batch_size = feature.shape[0] // 4
                i = np.random.randint(0, feature.shape[0])
                np.random.shuffle(old_class_list)
                lam = np.random.beta(0.5, 0.5)
                if lam > 0.6:
                    lam = lam * 0.6
                if np.random.random() >= 0.5:
                    temp = (1 + lam) * self.prototype[old_class_list[0]] - lam * feature.detach().cpu().numpy()[i]
                else:
                    temp = (1 - lam) * self.prototype[old_class_list[0]] + lam * feature.detach().cpu().numpy()[i]

                proto_aug.append(temp)
                proto_aug_label.append(old_class_list[0])

            proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().to(self.device)
            proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).to(self.device).long()
            aug_preds = self.model.head(proto_aug)
            joint_aug_preds = self.model.orientation_class_head(proto_aug)

            agg_preds = 0
            agg_preds = agg_preds + joint_aug_preds[:, ::4]

            aug_distillation_loss = F.kl_div(F.log_softmax(aug_preds, 1),
                                             F.softmax(agg_preds.detach(), 1),
                                             reduction='batchmean')
            loss_protoAug = (F.cross_entropy(aug_preds / self.temp, proto_aug_label) +
                             F.cross_entropy(joint_aug_preds / self.temp, proto_aug_label * 4) +
                             aug_distillation_loss)
            return (single_preds, joint_preds, joint_loss + signle_loss + distillation_loss +
                                               self.protoaug_weight * loss_protoAug +
                                               self.kd_weight * loss_kd)
