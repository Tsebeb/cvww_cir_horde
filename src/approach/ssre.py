import copy
from argparse import ArgumentParser
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from networks.ssre_network import SSRE_Net
from utils import _get_unique_targets
from .incremental_learning import Inc_Learning_Appr


def filter_para(model, lr):
    return [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': lr}]


class Appr(Inc_Learning_Appr):
    """Class implementing the SSRE approach described in https://arxiv.org/pdf/2203.06359.pdf
    Original reference code is implemented here: https://github.com/zhukaii/SSRE
    """

    def __init__(self, model: SSRE_Net, device, nepochs, lr, lr_min, lr_factor, lr_patience, initial_batch_size,
                 protoaug_weight, temp, kd_weight, clipgrad=10000, momentum=0, wd=0,
                 fix_bn=False, eval_on_train=False, logger=None, exemplars_dataset=None, use_early_stopping=True):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   fix_bn, eval_on_train, logger, exemplars_dataset, use_early_stopping)
        # Parameters
        self.protoaug_weight = protoaug_weight
        self.kd_weight = kd_weight
        self.temp = temp
        self.initial_batch_size = initial_batch_size

        # State variables
        self.old_model: Optional[SSRE_Net] = None
        self.prototype = None
        self.class_label = None
        self.classes_this_exp = []
        self.known_classes = []
        self.new_classes = []

    def _get_optimizer(self):
        optim_para = filter_para(self.model, self.lr[0])
        return torch.optim.Adam(optim_para, weight_decay=self.wd)

    def _get_classes_previous_learned(self):
        known_classes_set = set(self.known_classes)
        this_exp_classes = set(self.classes_this_exp)
        return list(known_classes_set.difference(this_exp_classes))

    @staticmethod
    def get_model_class():
        return SSRE_Net

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument("--initial-batch-size", default=64, type=int, help="SSRE initial batch size for the ")
        parser.add_argument("--protoaug-weight", default=10.0, type=float, help="protoAug loss weight")
        parser.add_argument("--kd-weight", default=1.0, type=float, help="knowledge distillation loss weight")
        parser.add_argument("--temp", default=0.1, type=float, help="training time temperature")
        return parser.parse_known_args(args)

    def _get_total_classes(self):
        return len(self.new_classes) + len(self.known_classes)

    def pre_train_process(self, t, trn_loader, val_loader):
        super().pre_train_process(t, trn_loader, val_loader)
        # Calculate number of classes here and a mapping from class_idx to indices in the dataset
        self.classes_this_exp = _get_unique_targets(trn_loader.dataset).tolist()
        for class_idx in self.classes_this_exp:
            if class_idx not in self.known_classes:
                self.new_classes.append(class_idx)

        self.model.modify_head(self._get_total_classes())
        self.model.to(self.device)
        if t > 0:
            self.model.network_expansion()

    def train_epoch(self, t, trn_loader: DataLoader):
        """Runs a single epoch"""
        if t == 0:
            trn_loader = DataLoader(trn_loader.dataset, batch_size=self.initial_batch_size,
                                    num_workers=trn_loader.num_workers, shuffle=True, pin_memory=trn_loader.pin_memory)

        self.model.train()
        if t > 0:
            self.model.freeze_bn()

        running_loss = 0.0
        running_acc = 0
        running_elements = 0
        for images, targets in trn_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            pred, loss = self._compute_loss(images, targets, trn_loader.batch_size)
            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

            # Update statistics
            running_loss += loss.item() * targets.size(0)
            running_acc += torch.sum(torch.argmax(pred, dim=1) == targets).cpu().item()
            running_elements += targets.size(0)
        if running_elements > 0:
            print(" | Train running: loss={:.3f}, acc={:6.2f}% |".format(running_loss / running_elements, running_acc / running_elements * 100), end="")

    def eval_early_stopping(self, t, val_loader):
        with torch.inference_mode():
            self.model.eval()

            val_loss = 0.0
            val_hits = 0
            num_hits = 0
            for images, targets in val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                pred, loss = self._compute_loss(images, targets, val_loader.batch_size)

                # Update statistics
                val_loss += loss.item() * targets.size(0)
                val_hits += torch.sum(torch.argmax(pred, dim=1) == targets).cpu().item()
                num_hits += targets.size(0)

            return val_loss / num_hits, val_hits / num_hits
    def post_train_process(self, t, trn_loader, val_loader):
        self.protoSave(t, trn_loader)
        if t > 0:
            self.model.network_compression()
        super().post_train_process(t, trn_loader, val_loader)
        # Reset old and new classes
        self.known_classes += self.new_classes
        self.new_classes = []

        self.old_model = deepcopy(self.model)
        self.old_model.cuda()
        self.old_model.eval()

    def _compute_loss(self, imgs, target, batch_size):
        if self.old_model is None:
            output = self.model(imgs)
            loss_cls = torch.nn.functional.cross_entropy(output / self.temp, target)
            return output, loss_cls
        else:
            feature = self.model.get_features(imgs)
            with torch.no_grad():
                feature_old = self.old_model.get_features(imgs)

            proto = torch.from_numpy(np.array(self.prototype)).t().cuda()
            proto_nor = torch.nn.functional.normalize(proto, p=2, dim=0, eps=1e-12)
            feature_nor = torch.nn.functional.normalize(feature, p=2, dim=-1, eps=1e-12)
            cos_dist = feature_nor @ proto_nor
            cos_dist = torch.max(cos_dist, dim=-1).values
            cos_dist2 = 1 - cos_dist
            output = self.model(imgs)
            loss_cls = torch.nn.functional.cross_entropy(output / self.temp, target, reduce=False)
            loss_cls = torch.mean(loss_cls * cos_dist2, dim=0)

            loss_kd = torch.norm(feature - feature_old, p=2, dim=1)
            loss_kd = torch.sum(loss_kd * cos_dist, dim=0)

            proto_aug = []
            proto_aug_label = []
            index = self._get_classes_previous_learned()
            for _ in range(batch_size):
                np.random.shuffle(index)
                temp = self.prototype[index[0]]
                proto_aug.append(temp)
                proto_aug_label.append(self.class_label[index[0]])

            proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().cuda()
            proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).cuda()
            soft_feat_aug = self.model.head(proto_aug)
            loss_protoAug = torch.nn.functional.cross_entropy(soft_feat_aug / self.temp, proto_aug_label)
            return output, loss_cls + self.protoaug_weight * loss_protoAug + self.kd_weight * loss_kd

    def protoSave(self, t, trn_loader):
        # if current_task > 0:
        features = []
        labels = []
        self.model.eval()
        with torch.no_grad():
            for i, (images, target) in enumerate(trn_loader):
                feature = self.model.get_features(images.cuda())
                if feature.shape[0] == trn_loader.batch_size:
                    labels.append(target.numpy())
                    features.append(feature.cpu().numpy())
        labels_set = np.unique(labels)
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        features = np.array(features)
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))

        prototype = []
        class_label = []
        for item in labels_set:
            index = np.where(item == labels)[0]
            class_label.append(item)
            feature_classwise = features[index]
            prototype.append(np.mean(feature_classwise, axis=0))

        if t == 0:
            self.prototype = prototype
            self.class_label = class_label
        else:
            self.prototype = np.concatenate((prototype, self.prototype), axis=0)
            self.class_label = np.concatenate((class_label, self.class_label), axis=0)
