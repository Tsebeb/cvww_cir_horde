import itertools
from copy import deepcopy

import numpy as np
import torch
from argparse import ArgumentParser

from torch.utils.data import DataLoader
from networks.il2a_network import IL2ANet
from utils import _get_unique_targets
from .incremental_learning import Inc_Learning_Appr


class Appr(Inc_Learning_Appr):
    """Class implementing the IL2A approach published at the NeurIPS 2021
    described in https://proceedings.neurips.cc/paper/2021/file/77ee3bc58ce560b86c2b59363281e914-Paper.pdf
    Original reference code is implemented here: https://github.com/Impression2805/IL2A
    """
    def __init__(self, model: IL2ANet, device, nepochs, lr, lr_min, lr_factor, lr_patience, seman_weight, kd_weight,
                 temp, class_aug_alpha, class_aug_num_mixups, seman_aug_ratio, clipgrad=10000, momentum=0, wd=0,
                 fix_bn=False, eval_on_train=False, logger=None,
                 exemplars_dataset=None, use_early_stopping=None):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   fix_bn, eval_on_train, logger, exemplars_dataset, use_early_stopping)
        # Parameters
        self.seman_weight = seman_weight
        self.kd_weight = kd_weight
        self.temp = temp
        self.class_aug_alpha = class_aug_alpha
        self.class_aug_num_mixups = class_aug_num_mixups
        self.sem_aug_ratio = seman_aug_ratio

        # State variables
        self.augnumclass = 0
        self.aug_index_mapping = None
        self.cov = None
        self.prototype = None
        self.class_label = None
        self.old_model = None
        self.classes_this_exp = []
        self.known_classes = []
        self.new_classes = []

    @staticmethod
    def get_model_class():
        return IL2ANet

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument("--seman-weight", default=10.0, type=float, help="protoAug loss weight")
        parser.add_argument("--kd-weight", default=10.0, type=float, help="knowledge distillation loss weight")
        parser.add_argument("--temp", default=0.1, type=float, help="training time temperature")

        parser.add_argument("--class-aug-alpha", default=20.0, type=float, help="")
        parser.add_argument("--class-aug-num-mixups", default=4, type=int, help="Number of feature combinations that should be produced")
        parser.add_argument("--seman-aug-ratio", default=2.5, type=float)
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
        # Calculate number of classes here and a mapping from class_idx to indices in the dataset
        self.classes_this_exp = sorted(_get_unique_targets(trn_loader.dataset).tolist())
        for class_idx in self.classes_this_exp:
            if class_idx not in self.known_classes:
                self.new_classes.append(class_idx)

        self.augnumclass = int(len(self.classes_this_exp) * (len(self.classes_this_exp) - 1) / 2)
        self.aug_index_mapping = {}
        for i, combination in enumerate(itertools.combinations(self.classes_this_exp, r=2)):
            if combination[0] not in self.aug_index_mapping:
                self.aug_index_mapping[combination[0]] = {}
            self.aug_index_mapping[combination[0]][combination[1]] = i
        self.model.modify_head(self._get_total_classes() + self.augnumclass)
        self.model.to(self.device)

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        running_loss = 0.0
        running_acc = 0
        running_elements = 0
        for images, targets in trn_loader:
            # Forward current model
            outputs, loss, tgets = self._compute_loss(images, targets.to(self.device), trn_loader.batch_size)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update statistics
            running_loss += loss.item() * targets.size(0)
            running_acc += torch.sum(torch.argmax(outputs, dim=1) == tgets.to(self.device)).cpu().item()
            running_elements += tgets.size(0)
        print(" | Train running: loss={:.3f}, acc={:6.2f}% |".format(running_loss / running_elements, running_acc / running_elements * 100), end="")

    def eval_early_stopping(self, t, val_loader):
        with torch.inference_mode():
            self.model.eval()
            hits, num_elements = 0, 0
            val_loss = 0.0
            for images, targets in val_loader:
                pred, loss, targets = self._compute_loss(images.to(self.device), targets.to(self.device), val_loader.batch_size)
                val_loss += loss.detach().item() * targets.size(0)
                hits += torch.sum(torch.argmax(pred, dim=1) == targets.to(self.device)).cpu().item()
                num_elements += targets.size(0)

            return val_loss / num_elements, hits / num_elements

    def post_train_process(self, t, trn_loader, val_loader):
        self.protoSave(t, trn_loader)
        super().post_train_process(t, trn_loader, val_loader)

        # Reset old and new classes
        self.known_classes += self.new_classes
        self.new_classes = []

        self.model.shrink_head(len(self.known_classes))
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

        prototype = []
        cov = []
        class_label = []
        for item in labels_set:
            index = np.where(item == labels)[0]
            class_label.append(item)
            feature_classwise = features[index]
            prototype.append(np.mean(feature_classwise, axis=0))
            cov_class = np.cov(feature_classwise.T)
            cov.append(cov_class)

        if t == 0:
            self.cov = np.concatenate(cov, axis=0).reshape([-1, self.model.out_size, self.model.out_size])
            self.prototype = prototype
            self.class_label = class_label
        else:
            self.cov = np.concatenate((cov, self.cov), axis=0)
            self.prototype = np.concatenate((prototype, self.prototype), axis=0)
            self.class_label = np.concatenate((class_label, self.class_label), axis=0)

    def semanAug(self, features, y, labels, ratio):
        """Copied from original reference implementation: https://github.com/Impression2805/IL2A/blob/main/cifar/IL2A.py#L151 """
        N = features.size(0)
        C = self._get_total_classes()
        A = features.size(1)
        weight_m = list(self.model.head.parameters())[0]
        weight_m = weight_m[:self._get_total_classes(), :]
        NxW_ij = weight_m.expand(N, C, A)
        NxW_kj = torch.gather(NxW_ij, 1, labels.view(N, 1, 1).expand(N, C, A))
        CV = self.cov
        labels = labels.cpu()
        CV_temp = torch.from_numpy(CV[labels]).to(self.device)
        sigma2 = ratio * torch.bmm(torch.bmm(NxW_ij - NxW_kj, CV_temp.float()), (NxW_ij - NxW_kj).permute(0, 2, 1))
        sigma2 = sigma2.mul(torch.eye(C).to(self.device).expand(N, C, C)).sum(2).view(N, C)
        aug_result = y + 0.5 * sigma2
        return aug_result

    @torch.no_grad()
    def classAug(self, x, y):  # mixup based
        """Copied from original reference implementation: https://github.com/Impression2805/IL2A/blob/main/cifar/IL2A.py#L151 """
        batch_size = x.size()[0]
        mix_data = []
        mix_target = []
        for _ in range(self.class_aug_num_mixups):
            index = torch.randperm(batch_size).to(self.device)
            for i in range(batch_size):
                if y[i] != y[index][i]:
                    new_label = self.generate_label(y[i].item(), y[index][i].item())
                    lam = np.random.beta(self.class_aug_alpha, self.class_aug_alpha)
                    if lam < 0.4 or lam > 0.6:
                        lam = 0.5
                    mix_data.append(lam * x[i] + (1 - lam) * x[index, :][i])
                    mix_target.append(new_label)

        new_target = torch.Tensor(mix_target)
        y = torch.cat((y, new_target.to(self.device).long()), 0)
        for item in mix_data:
            x = torch.cat((x, item.unsqueeze(0)), 0)
        return x, y

    def generate_label(self, y_a, y_b):
        y_a, y_b = y_a, y_b
        assert y_a != y_b
        if y_a > y_b:
            tmp = y_a
            y_a = y_b
            y_b = tmp
        return self.aug_index_mapping[y_a][y_b] + self._get_total_classes()

    def _compute_loss(self, imgs, target, batch_size):
        imgs, target = imgs.to(self.device), target.to(self.device)
        imgs, target = self.classAug(imgs, target)
        output = self.model(imgs)
        loss_cls = torch.nn.functional.cross_entropy(output / self.temp, target)

        if self.old_model is None:
            return output, loss_cls, target
        else:
            feature = self.model.get_features(imgs)
            feature_old = self.old_model.get_features(imgs)
            loss_kd = torch.dist(feature, feature_old, 2)

            proto_aug = []
            proto_aug_label = []
            index = self._get_classes_previous_learned()
            if len(index) == 0:
                return output, loss_cls, target

            for _ in range(batch_size):
                np.random.shuffle(index)
                temp = self.prototype[index[0]]
                proto_aug.append(temp)
                proto_aug_label.append(self.class_label[index[0]])

            proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().to(self.device)
            proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).to(self.device)
            soft_feat_aug = self.model.head(proto_aug)
            soft_feat_aug = soft_feat_aug[:, :self._get_total_classes()]

            isda_aug_proto_aug = self.semanAug(proto_aug, soft_feat_aug, proto_aug_label, self.sem_aug_ratio)
            loss_semanAug = torch.nn.functional.cross_entropy(isda_aug_proto_aug / self.temp, proto_aug_label)
            return output, loss_cls + self.seman_weight * loss_semanAug + self.kd_weight * loss_kd, target
