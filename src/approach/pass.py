import time
from copy import deepcopy

import numpy as np
import torch
from argparse import ArgumentParser

from torch.utils.data import DataLoader
from networks.pass_network import PASS_Net
from utils import _get_unique_targets
from .incremental_learning import Inc_Learning_Appr


class Appr(Inc_Learning_Appr):
    """Class implementing the PASS (prototype augmentation and self-supervision) approach published at the CVPR 2021
    described in https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Prototype_Augmentation_and_Self-Supervision_for_Incremental_Learning_CVPR_2021_paper.pdf
    Original reference code is implemented here: https://github.com/Impression2805/CVPR21_PASS
    """

    # FIXME: Discuss with Marc we create 4 rotated augmentation of each image and try to predict both class and augmentation / rotation
    # Predictions are then handled in both class and orientation -> however source code has a random flipping transformation -> messes up the combination not handled?

    def __init__(self, model: PASS_Net, device, nepochs, lr, lr_min, lr_factor, lr_patience, protoaug_weight, kd_weight,
                 temp, clipgrad=10000, momentum=0, wd=0, fix_bn=False, eval_on_train=False, logger=None, exemplars_dataset=None,
                 use_early_stopping=True):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   fix_bn, eval_on_train, logger, exemplars_dataset, use_early_stopping)
        # Parameters
        self.protoaug_weight = protoaug_weight
        self.kd_weight = kd_weight
        self.temp = temp

        # State variables
        self.radius = 0
        self.prototype = None
        self.class_label = None
        self.old_model = None
        self.known_classes = []
        self.new_classes = []
        self.classes_this_exp = []

    @staticmethod
    def get_model_class():
        return PASS_Net

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument("--protoaug-weight", default=10.0, type=float, help="protoAug loss weight")
        parser.add_argument("--kd-weight", default=10.0, type=float, help="knowledge distillation loss weight")
        parser.add_argument("--temp", default=0.1, type=float, help="training time temperature")
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

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()

        running_loss = 0.0
        running_acc = 0
        running_elements = 0
        for images, targets in trn_loader:
            orig_channels, orig_height, orig_width = images.size(1), images.size(2), images.size(3)
            # self-supervised learning based label augmentation
            images = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1)
            images = images.view(-1, orig_channels, orig_height, orig_width)
            targets = torch.stack([targets * 4 + k for k in range(4)], 1).view(-1)
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs, loss = self._compute_loss(images, targets)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

            # Update statistics
            running_loss += loss.item() * targets.size(0)
            running_acc += torch.sum(torch.argmax(outputs, dim=1) == targets).cpu().item()
            running_elements += targets.size(0)
        print(" | Train running: loss={:.3f}, acc={:6.2f}% |".format(running_loss / running_elements, running_acc / running_elements * 100), end="")

    def eval_early_stopping(self, t, val_loader):
        with torch.inference_mode():
            self.model.eval()
            hits, num_elements = 0, 0
            val_loss = 0.0

            for images, targets in val_loader:
                orig_channels, orig_height, orig_width = images.size(1), images.size(2), images.size(3)
                # self-supervised learning based label augmentation
                images = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1)
                images = images.view(-1, orig_channels, orig_height, orig_width)
                targets = torch.stack([targets * 4 + k for k in range(4)], 1).view(-1)
                images = images.to(self.device)
                targets = targets.to(self.device)
                outputs, loss = self._compute_loss(images, targets)

                hits += torch.sum(torch.argmax(outputs, dim=1) == targets).cpu().item()
                num_elements += targets.size(0)
                val_loss += loss.item() * targets.size(0)
            return val_loss / num_elements, hits / num_elements

    def post_train_process(self, t, trn_loader, val_loader):
        self.protoSave(t, trn_loader)
        super().post_train_process(t, trn_loader, val_loader)
        # Reset old and new classes
        self.known_classes += self.new_classes
        self.new_classes = []

        # Updates the normal head so that we only have the 4 base classes left!
        self.model.collapse_orientation_head()
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
                if trn_loader.batch_size == target.size(0):
                    feature = self.model.get_features(images.to(self.device))
                    labels.append(target.numpy())
                    features.append(feature.cpu().numpy())

        labels_set = np.unique(labels)
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        features = np.array(features)
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))
        feature_dim = features.shape[1]

        prototype = []
        radius = []
        class_label = []
        for item in labels_set:
            index = np.where(item == labels)[0]
            class_label.append(item)
            feature_classwise = features[index]
            prototype.append(np.mean(feature_classwise, axis=0))
            if t == 0:
                cov = np.cov(feature_classwise.T)
                radius.append(np.trace(cov) / feature_dim)

        if t == 0:
            self.radius = np.sqrt(np.mean(radius))
            self.prototype = prototype
            self.class_label = class_label
            print(self.radius)
        else:
            self.prototype = np.concatenate((prototype, self.prototype), axis=0)
            self.class_label = np.concatenate((class_label, self.class_label), axis=0)

    def _compute_loss(self, imgs, target):
        output = self.model.orientation_forward(imgs)
        output, target = output.to(self.device), target.to(self.device)
        loss_cls = torch.nn.functional.cross_entropy(output / self.temp, target)
        if self.old_model is None:
            return output, loss_cls
        else:
            feature = self.model.get_features(imgs)
            feature_old = self.old_model.get_features(imgs)
            loss_kd = torch.dist(feature, feature_old, 2)

            proto_aug = []
            proto_aug_label = []
            index = self._get_classes_previous_learned()
            for _ in range(target.size(0)):
                np.random.shuffle(index)
                temp = self.prototype[index[0]] + np.random.normal(0, 1, self.model.out_size) * self.radius
                proto_aug.append(temp)
                proto_aug_label.append(4 * self.class_label[index[0]])

            proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().to(self.device)
            proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).to(self.device)
            soft_feat_aug = self.model.orientation_class_head(proto_aug)
            loss_protoAug = torch.nn.functional.cross_entropy(soft_feat_aug / self.temp, proto_aug_label)

            return output, loss_cls + self.protoaug_weight * loss_protoAug + self.kd_weight * loss_kd
