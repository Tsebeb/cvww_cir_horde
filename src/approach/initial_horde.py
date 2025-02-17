import copy
import itertools
import math
import time
import warnings
from copy import deepcopy
from itertools import combinations
from random import shuffle
from typing import Optional, Tuple, List

import numpy as np
import torch
from argparse import ArgumentParser

from torch.nn import Linear
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, SubsetRandomSampler

from horde_utils import convert_class_prototypes, calculate_partial_class_prototype
from networks.horde_network import HordeModel
from networks.init_horde_network import InitHordeModel
from utils import get_confusion_matrix
from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


def pdist(vec):
    return -2 * vec.mm(torch.t(vec)) + vec.pow(2).sum(dim=1).view(1, -1) + vec.pow(2).sum(dim=1).view(-1, 1)


def _set_learning_rate(optimizer: Optimizer, learning_rate):
    for g in optimizer.param_groups:
        g['lr'] = learning_rate


# Pair Selectors for online contrastive loss calculations
class AllPositivePairSelector:
    """
    Discards embeddings and generates all possible pairs given labels.
    If balance is True, negative pairs are a random sample to match the number of positive samples
    Adapted from https://github.com/adambielski/siamese-triplet
    """
    def __init__(self, balance=True):
        self.balance = balance

    def get_pairs(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]]
        negative_pairs = all_pairs[labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]]
        if self.balance:
            # if more positive than negative pairs return all negatives that are available
            if len(positive_pairs) >= len(negative_pairs):
                return positive_pairs, negative_pairs
            negative_pairs = negative_pairs[torch.randperm(len(negative_pairs))[:len(positive_pairs)]]

        return positive_pairs, negative_pairs


class HardNegativePairSelector:
    """
    Creates all possible positive pairs. For negative pairs, pairs with smallest distance are taken into consideration,
    matching the number of positive pairs.
    """
    def __init__(self, cpu=True):
        self.cpu = cpu

    def get_pairs(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)

        labels = labels.cpu().data.numpy()
        combis = list(combinations(range(len(labels)), 2))
        if len(combis) == 0:
            return [], []
        all_pairs = torch.LongTensor(np.array(combis))
        positive_pairs = all_pairs[labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]]
        negative_pairs = all_pairs[labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]]

        negative_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
        negative_distances = negative_distances.cpu().data.numpy()
        # if more positive than negative pairs return all negatives that are available
        if len(positive_pairs) >= len(negative_distances):
            return positive_pairs, negative_pairs
        top_negatives = np.argpartition(negative_distances, len(positive_pairs))[:len(positive_pairs)]
        top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        return positive_pairs, top_negative_pairs


# noinspection PyTypeChecker
class Appr(Inc_Learning_Appr):
    """Class implementing the finetuning baseline"""

    def __init__(self, model: HordeModel, device, training_method, use_adaptive_alpha, alpha, ml_dims, fe_lr, fe_epochs,
                 ignore_unhelpful_train, ml_margin, ml_pair_selector, acc_thr, project_unk_mean, project_unk_std,
                 use_curr_cls_ph2, num_sim_feats, num_iterations_for_mean, fe_selection: str, fe_conf_strong_factor: float,
                 min_fe_improve_score_per_class: float, use_lwf_fe: bool, fe_lwf_lamb: float, fe_lwf_T: float, warmup_epochs: int,
                 use_self_supervision, fe_dist_overlap_min_class_overlap, acc_prototype: bool,
                 nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000, momentum=0, wd=0,
                 fix_bn=False, eval_on_train=False, logger=None, exemplars_dataset=None, use_early_stopping=None,):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   fix_bn, eval_on_train, logger, exemplars_dataset, use_early_stopping)
        self.single_fe_latent_dim = self.model.out_size
        self._known_classes = []
        self.num_fe = model.number_of_feature_extractors
        self.acc_thr = acc_thr
        self.training_method_fe = training_method

        # training FE parameters
        self.use_adaptive_alpha = use_adaptive_alpha
        self.alpha = alpha
        self.ml_dims = ml_dims
        self.ml_margin = ml_margin
        self.fe_lr = fe_lr
        self.fe_epochs = fe_epochs
        self._project_unk_mean = project_unk_mean
        self.use_self_supervision = use_self_supervision

        self.use_lwf_fe = use_lwf_fe
        self.fe_lwf_lamb = fe_lwf_lamb
        self.fe_lwf_T = fe_lwf_T

        self._backup_model = None
        self.ignore_unhelpful_train = ignore_unhelpful_train
        self.warmup_epochs = warmup_epochs

        # State variables for horde algorithm
        self.embedding_size = self.num_fe * self.single_fe_latent_dim
        self.initial_std_embedding = torch.ones((len(self._known_classes), self.model.init_out_size), device=self.device)
        self.initial_mean_embedding = torch.zeros((len(self._known_classes), self.model.init_out_size), device=self.device)
        self.std_embedding = torch.ones((len(self._known_classes), self.num_fe, self.single_fe_latent_dim), device=self.device)
        self.mean_embedding = torch.zeros((len(self._known_classes), self.num_fe, self.single_fe_latent_dim), device=self.device)
        self.__project_unknown_std = project_unk_std

        self.initial_max_samples_seen = torch.zeros((len(self._known_classes)))
        self.max_samples_seen = torch.zeros((len(self._known_classes), self.num_fe))
        self.cls_initial_trained = torch.zeros((len(self._known_classes)))
        self.cls_trn_in_fe = torch.zeros((len(self._known_classes), self.num_fe))
        self._use_curr_cls_ph2 = use_curr_cls_ph2
        self._num_sim_feats = num_sim_feats
        self.__num_iterations_for_mean = num_iterations_for_mean

        if ml_pair_selector == "hard_negative":
            self._pair_selector = HardNegativePairSelector()
        elif ml_pair_selector == "all_positive":
            self._pair_selector = AllPositivePairSelector()
        else:
            raise RuntimeError("Unsupported Pair selector")
        self.__clip_std_min = -2.0
        self.__clip_std_max = 2.0

        self._cur_val_to_idx_dict: Optional[dict] = None
        self._initial_training_done: bool = False
        assert len(fe_lr) == len(fe_epochs)

        # Arguments and members for the feature selection
        self.fe_selection = fe_selection
        ###### ARGUMENTS FOR CONFUSION MATRIX OTHERWISE UNUSED! ######
        self._fe_selection_score = torch.zeros((self.num_fe, ))
        self._fe_conf_strong_factor = fe_conf_strong_factor
        self._fe_possible_max_score = None
        self._min_fe_improve_score_per_class = min_fe_improve_score_per_class
        self._fe_dist_overlap_min_class_overlap = fe_dist_overlap_min_class_overlap
        self.acc_prototype = acc_prototype

    @staticmethod
    def get_model_class():
        return InitHordeModel

    def _increase_buffer_size(self, new_size: int):
        old_size = len(self._known_classes)

        cls_trn_in_fe_new = torch.zeros(((new_size, self.num_fe)))
        cls_trn_in_fe_new[:old_size, :] = self.cls_trn_in_fe[:, :]  # copy old data
        self.cls_trn_in_fe = cls_trn_in_fe_new

        cls_initial_trained = torch.zeros((new_size))
        cls_initial_trained[:old_size] = self.cls_initial_trained[:]  # copy old data
        self.cls_initial_trained = cls_initial_trained

        # Feature Extractor Values
        max_samples_seen_new = torch.zeros(((new_size, self.num_fe)))
        max_samples_seen_new[:old_size, :] = self.max_samples_seen[:, :] # copy old data
        self.max_samples_seen = max_samples_seen_new
        std_embedding_new = torch.ones((new_size, self.num_fe, self.single_fe_latent_dim), device=self.device)
        std_embedding_new[:old_size, :, :] = self.std_embedding[:, :, :]  # copy old data
        self.std_embedding = std_embedding_new
        mean_embedding_new = torch.zeros((new_size, self.num_fe, self.single_fe_latent_dim), device=self.device)
        mean_embedding_new[:old_size, :, :] = self.mean_embedding[:, :, :]  # copy old data
        self.mean_embedding = mean_embedding_new

        # Initial Values
        initial_samples_seen = torch.zeros((new_size))
        initial_samples_seen[:old_size] = self.initial_max_samples_seen[:]
        self.initial_max_samples_seen = initial_samples_seen
        initial_std_embedding_new = torch.ones((new_size, self.model.init_out_size), device=self.device)
        initial_std_embedding_new[:old_size, :] = self.initial_std_embedding[:, :]  # copy old data
        self.initial_std_embedding = initial_std_embedding_new
        initial_embedding_mean = torch.zeros((new_size, self.model.init_out_size), device=self.device)
        initial_embedding_mean[:old_size, :] = self.initial_mean_embedding[:, :]  # copy old data
        self.initial_mean_embedding = initial_embedding_mean

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument("--ignore-unhelpful-train", action="store_true", help="")
        parser.add_argument("--acc-thr", type=float, default=1.0)
        parser.add_argument("--project-unk-mean", type=str, default="feats", choices=["feats", "zeros", "noise"])
        parser.add_argument("--project_unk_std", type=float, default=1.0)
        parser.add_argument("--use-curr-cls-ph2", action="store_true")
        parser.add_argument("--num-sim-feats", type=int, default=1)
        parser.add_argument("--num-iterations-for-mean", type=int, default=1, help="Number of iterations over training dataset ")

        # Arguments for Feature Extractor Trainings
        parser.add_argument('--training-method', type=str, default="ce_ml", choices=["ce", "ml", "ce_ml"],
                            required=False,  help='Number of feature extractors used for the backbone of the ')
        parser.add_argument('--use-adaptive-alpha', action="store_true", help="")
        parser.add_argument("--alpha", type=float, default=0.5, help="only used if use_adaptive")
        parser.add_argument("--fe-lr", type=float, default=[0.001], nargs="+")
        parser.add_argument("--fe-epochs", type=int, default=[70], nargs="+")
        parser.add_argument("--ml-dims", type=int, default=128)
        parser.add_argument("--ml-pair-selector", default="hard_negative", choices=["hard_negative", "all_positive"])
        parser.add_argument("--ml-margin", type=float, default=0.5)
        parser.add_argument("--use-self-supervision", action="store_true", help="Add Self-Supervision like PASS for feature extractor training")

        # Arguments for LwF for the feature extractor base
        parser.add_argument("--use-lwf-fe", action="store_true")
        parser.add_argument('--fe-lwf-lamb', default=0.5, type=float, required=False, help='Forgetting-intransigence trade-off (default=%(default)s)')
        parser.add_argument('--fe-lwf-T', default=2, type=int, required=False, help='Temperature scaling (default=%(default)s)')

        parser.add_argument("--warmup-epochs", type=int, default=5, help="Utilize warmup period")
        # version of feature selection training
        parser.add_argument("--fe-selection", type=str, default="max_classes", choices=["challenge", "confusion_matrix", "distribution_overlap", "max_classes"])
        parser.add_argument("--fe-conf-strong-factor", type=float, default=2.0, help="Counts the score x times if both the class that it"
                                                                                     "is struggeling and the class that it is confused with present in the current task")
        parser.add_argument("--min-fe-improve-score-per-class", type=float, default=0.15, help="The minimum class score that should be improvable for a new feature extractor to be trained")
        parser.add_argument("--fe-dist-overlap-min-class-overlap", type=float, default=0.1, help="The minimum average amount of a class to overlap")
        parser.add_argument("--acc-prototype", action="store_true", help="accumulate prototypes when repeated")
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        params = self.model.linear.parameters()
        return torch.optim.Adam(params, lr=self.lr[0], weight_decay=self.wd)

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
            if class_idx not in self._known_classes:
                new_classes.append(class_idx)

        if len(new_classes) > 0:
            self._increase_buffer_size(len(self._known_classes) + len(new_classes))
            self._known_classes += new_classes

        super(Appr, self).pre_train_process(t, trn_loader, val_loader)

    def _decide_if_new_fe_is_trained(self, trn_loader) -> Tuple[bool, int]:
        """
        Returns whether a feature extractor should be trained and if necessary which fe should be replaced
        """
        # If initial training is not done always train an initial fe
        if not self._initial_training_done:
            return True, 0

        if self.fe_selection == "challenge":
            # Decide if we want to build a feature extractor for this experience or not
            seen_cls_fe = self._get_cls_with_mean()
            trained_cls_fe = self._get_cls_fe_trained()
            new_untrained_fe_cls = [cls for cls in self._cur_val_to_idx_dict.keys() if trained_cls_fe[cls] == 0]
            should_be_trained = seen_cls_fe.sum() < 85 and len(new_untrained_fe_cls) >= 5

            if should_be_trained:
                # If Model has room or check wether a fe should be replaced
                if self.model.has_room_to_grow():
                    return should_be_trained, len(self.model.feature_extractors)
                else:
                    # Test whether there is a feature extractor with less classes than before
                    min_fe_cls, min_fe_pos = self.cls_trn_in_fe.sum(dim=0).min(dim=0)
                    if len(self._cur_val_to_idx_dict.keys()) > min_fe_cls:
                        return True, min_fe_pos
                    else:
                        return False, -1
            else:
                return False, -1
        elif self.fe_selection == "confusion_matrix":
            self._fe_possible_max_score = self._get_fe_adjusted_conf_score(trn_loader)
            print("\nMax Possible Improvement Score ", f"{self._fe_possible_max_score.item():.2f}", "classes",
                  len(self._cur_val_to_idx_dict.keys()), "train_score", len(self._cur_val_to_idx_dict.keys()) * self._min_fe_improve_score_per_class)
            print("Scores of other feature selectors ", self._fe_selection_score.tolist())
            if self._fe_possible_max_score > self._min_fe_improve_score_per_class * len(self._cur_val_to_idx_dict.keys()) \
                    and len(self._cur_val_to_idx_dict) > 1:
                # Build up FE if room is there
                if self.model.has_room_to_grow():
                    return True, len(self.model.feature_extractors)
                else:
                    # compare possible max score with others
                    min_score = torch.min(self._fe_selection_score) if self._fe_selection_score.dim != 0 else self._fe_selection_score
                    min_pos = torch.argmin(self._fe_selection_score).item() if self._fe_selection_score.dim != 0 else 0
                    if self._fe_possible_max_score > min_score:
                        return True, min_pos
            return False, -1
        elif self.fe_selection == "max_classes":
            classes_initial, classes_per_fe = self._get_list_of_classes_represented_by_fe()
            total_current = copy.deepcopy(classes_initial)
            for class_set in classes_per_fe:
                total_current.update(class_set)
            classes_current_task = set(self._cur_val_to_idx_dict.keys())

            # Finish THIS !
            best_variation = len(total_current)
            train_fe = False
            train_idx = -1
            for fe_idx, classes_fe_current in enumerate(classes_per_fe):
                fe_replaced_conf = copy.deepcopy(classes_initial)
                fe_replaced_conf.update(classes_current_task)
                for i, class_set in enumerate(classes_per_fe):
                    if fe_idx != i:
                        fe_replaced_conf.update(class_set)

                if best_variation < len(fe_replaced_conf):
                    best_variation = len(fe_replaced_conf)
                    train_fe = True
                    train_idx = fe_idx
            return train_fe, train_idx
        else:
            raise NotImplementedError("The given fe selection method")

    def _get_fe_adjusted_conf_score(self, trn_loader: DataLoader):
        conf_mat = get_confusion_matrix(self.model, trn_loader, len(self._known_classes), self.device).clone().float()
        # print_confusion_matrix(conf_mat)
        # double the importance of all class combinations
        for class_combination in itertools.combinations(self._cur_val_to_idx_dict.keys(), 2):
            conf_mat[class_combination[0], class_combination[1]] *= self._fe_conf_strong_factor
            conf_mat[class_combination[1], class_combination[0]] *= self._fe_conf_strong_factor

        total = torch.sum(conf_mat, dim=1)
        mistakes_ratio = torch.nan_to_num((total - torch.diag(conf_mat)) / total, 0.0)
        return torch.sum(mistakes_ratio)

    def _update_fe_selection_metrics_post_training(self, trn_loader: DataLoader, feature_extractor_trained: bool, fe_idx: int):
        if self.fe_selection == "confusion_matrix" and feature_extractor_trained:
            score = self._get_fe_adjusted_conf_score(trn_loader)
            if not self._initial_training_done:
                self._fe_selection_score[fe_idx] = len(self._cur_val_to_idx_dict.keys()) - score
            else:
                self._fe_selection_score[fe_idx] = self._fe_possible_max_score - score  # store the score improvement
            print("Final Score: ", f"{self._fe_selection_score[fe_idx].item():.2f}", "Before Training",
                  f"{self._fe_selection_score[fe_idx] + score:.2f}", "After Training", f"{score:.2f}")
            self._fe_possible_max_score = 0.0

        # post training
        self._initial_training_done = True

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        # Calculate FZ for current data -- get performance and make deepcopy of model
        if self.ignore_unhelpful_train:
            self._backup_model = deepcopy(self.model)
            _acc_before = self._calculate_accuracy_score(trn_loader)

        # Train new FE on first exp or when fewer than 85 classes have been trained, and 5 of those are completely new
        should_train_fe, fe_idx = self._decide_if_new_fe_is_trained(trn_loader)
        if should_train_fe:
            print(f"Training new Feature Extractor for position {fe_idx}")
            feature_extractor = self._train_feature_extractor(t, fe_idx, trn_loader, val_loader)
            modify_ensemble, fe_idx = self._decide_if_fe_should_be_added(t, trn_loader, fe_idx, feature_extractor)
            if modify_ensemble:
                if not self._initial_training_done:
                    self.model.add_initial_feature_extractor(feature_extractor)
                    self.cls_initial_trained[list(self._cur_val_to_idx_dict.keys())] = 1
                else:
                    self.model.add_feature_extractor(feature_extractor, fe_idx)
                    self.max_samples_seen[:, fe_idx] = 0
                    self.cls_trn_in_fe[:,  fe_idx] = 0
                    self.cls_trn_in_fe[list(self._cur_val_to_idx_dict.keys()), fe_idx] = 1
        # Calculate Embeddings Now that the feature extractor is trained
        self._calculate_embedding_statistics(trn_loader)
        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)
        # Updates Scores and trackings for feature extractors and replacements
        self._update_fe_selection_metrics_post_training(trn_loader, should_train_fe, fe_idx)
        # Calculate performance of current data
        if self.ignore_unhelpful_train:
            _acc_after = self._calculate_accuracy_score(trn_loader)
            # If performance is not better than FZ from before training, recover old model
            if _acc_before >= _acc_after * self.acc_thr:
                self.model = deepcopy(self._backup_model)
                print(f"I've just ignored what I trained! Acc before {_acc_before} - Acc after {_acc_after}")
                # Delete old model
                self._backup_model = None

    def _get_cls_with_mean(self):
        return torch.clamp(self.initial_max_samples_seen, 0, 1)

    def _get_cls_fe_trained(self):
        return torch.clamp(self.cls_trn_in_fe.sum(dim=1), 0, 1)

    def _get_list_of_classes_represented_by_fe(self):
        initial_classes = set(torch.flatten(torch.nonzero(self.cls_initial_trained)).tolist())
        total = []
        for c in range(self.cls_trn_in_fe.size(1)):
            total.append(set(torch.flatten(torch.nonzero(self.cls_trn_in_fe[:, c])).tolist()))
        return initial_classes, total


    def _get_cls_mean(self, feats, class_idx):
        cls_mean = 0.01 * torch.randn((self.num_fe, self.single_fe_latent_dim)).to(self.device)
        for fe in range(len(self.model.feature_extractors)):
            if self.max_samples_seen[class_idx, fe] > 0:
                # When we have the mean calculated, we use it
                cls_mean[fe, :] = self.mean_embedding[class_idx, fe, :]
            else:
                if self._project_unk_mean == 'zeros':
                    cls_mean[fe, :] = 0.0
                elif self._project_unk_mean == 'feats':
                    cls_mean[fe, :] = feats[self.model.init_out_size + fe * self.single_fe_latent_dim : self.model.init_out_size + (fe + 1) * self.single_fe_latent_dim]
                elif self._project_unk_mean == 'noise':
                    # When random we just leave the already initialized random noise
                    pass
                else:
                    raise RuntimeError("Invalid project unknown mean method.")
        return torch.cat((self.initial_mean_embedding[class_idx, :], cls_mean.ravel()))

    def _get_cls_std(self, class_idx):
        cls_std = self.__project_unknown_std * torch.ones((self.num_fe, self.single_fe_latent_dim)).to(self.device)
        for fe in range(len(self.model.feature_extractors)):
            if self.max_samples_seen[class_idx, fe] > 0:
                # When we have the std calculated, we use it
                cls_std[fe, :] = self.std_embedding[class_idx, fe, :]
            # When the std has not been calculated, then set the base one
        return torch.cat((self.initial_std_embedding[class_idx, :], cls_std.ravel()))

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()

        running_loss = 0.0
        running_hits, num_elements = 0, 0

        for images, targets in trn_loader:
            # Forward current model
            outputs, features = self.model(images.to(self.device), return_features=True)
            loss = self.criterion(outputs, targets.to(self.device))

            # HORDE backprop part!
            horde_loss = torch.tensor(0.0).to(self.device)
            curr_exp_cls = self._cur_val_to_idx_dict.keys()
            for i, y in enumerate(targets):
                pseudo_feats, pseudo_targets = [], []
                # Extract label and features from current sample
                orig_cls = y.cpu().item()
                orig_feats = features[i].detach()
                # Choose new valid classes to simulate/hallucinate
                all_sim_cls = [idx for idx, valid in enumerate(self._get_cls_with_mean()) if valid and idx != orig_cls]
                # Remove the classes from the current experience
                if not self._use_curr_cls_ph2:
                    all_sim_cls = [elem for elem in all_sim_cls if elem not in curr_exp_cls]
                if len(all_sim_cls) > 0:
                    # Randomly choose as many simulated classes as needed
                    shuffle(all_sim_cls)
                    all_sim_cls = all_sim_cls[:min(self._num_sim_feats, len(all_sim_cls))]
                    # Simulate each new feature
                    for sim_cls in all_sim_cls:
                        # Calculate offset from original class
                        orig_cls_offset = orig_feats - self._get_cls_mean(orig_feats, orig_cls)  # remove orig mean
                        orig_cls_offset = orig_cls_offset / (self._get_cls_std(orig_cls) + 1e-12)  # project orig std
                        orig_cls_offset = torch.clamp(orig_cls_offset, self.__clip_std_min, self.__clip_std_max)
                        # project and shift with the simulated class mean and std
                        sim_feat = self._get_cls_mean(orig_feats, sim_cls) + orig_cls_offset * self._get_cls_std(sim_cls)
                        sim_feat = sim_feat.detach().unsqueeze_(0)
                        # Store additional class elements
                        pseudo_feats.append(sim_feat)
                        pseudo_targets.append(sim_cls)
                    # Stack pseudo-features and pass them through the head
                    pseudo_feats = torch.vstack(pseudo_feats)
                    pseudo_targets = torch.tensor(pseudo_targets).to(self.device)
                    output = self.model.linear(pseudo_feats)
                    pseudo_loss = torch.nn.functional.cross_entropy(output, pseudo_targets)
                    horde_loss += pseudo_loss

            # Normalize and add to the main training loss
            loss += horde_loss / targets.shape[0]

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

            running_loss += loss.item() * targets.size(0)
            running_hits += torch.sum(torch.argmax(outputs, dim=1) == targets.to(self.device)).cpu().item()
            num_elements += targets.size(0)
        print("| Train running: loss={:.3f}, acc={:6.2f}% |".format(running_loss / num_elements, running_hits / num_elements * 100), end="")

    def eval_early_stopping(self, t, val_loader):
        with torch.inference_mode():
            self.model.eval()
            val_loss = 0.0
            val_hits, val_num_elements = 0, 0

            for images, targets in val_loader:
                # Forward current model
                outputs, features = self.model(images.to(self.device), return_features=True)
                loss = self.criterion(outputs, targets.to(self.device))

                # HORDE backprop part!
                horde_loss = torch.tensor(0.0).to(self.device)
                curr_exp_cls = self._cur_val_to_idx_dict.keys()
                for i, y in enumerate(targets):
                    pseudo_feats, pseudo_targets = [], []
                    # Extract label and features from current sample
                    orig_cls = y.cpu().item()
                    orig_feats = features[i].detach()
                    # Choose new valid classes to simulate/hallucinate
                    all_sim_cls = [idx for idx, valid in enumerate(self._get_cls_with_mean()) if valid and idx != orig_cls]
                    # Remove the classes from the current experience
                    if not self._use_curr_cls_ph2:
                        all_sim_cls = [elem for elem in all_sim_cls if elem not in curr_exp_cls]
                    if len(all_sim_cls) > 0:
                        # Randomly choose as many simulated classes as needed
                        shuffle(all_sim_cls)
                        all_sim_cls = all_sim_cls[:min(self._num_sim_feats, len(all_sim_cls))]
                        # Simulate each new feature
                        for sim_cls in all_sim_cls:
                            # Calculate offset from original class
                            orig_cls_offset = orig_feats - self._get_cls_mean(orig_feats, orig_cls)  # remove orig mean
                            orig_cls_offset = orig_cls_offset / (self._get_cls_std(orig_cls) + 1e-8)  # project orig std
                            orig_cls_offset = torch.clamp(orig_cls_offset, self.__clip_std_min, self.__clip_std_max)
                            # project and shift with the simulated class mean and std
                            sim_feat = self._get_cls_mean(orig_feats, sim_cls) + orig_cls_offset * self._get_cls_std(
                                sim_cls)
                            sim_feat = sim_feat.detach().unsqueeze_(0)
                            # Store additional class elements
                            pseudo_feats.append(sim_feat)
                            pseudo_targets.append(sim_cls)
                        # Stack pseudo-features and pass them through the head
                        pseudo_feats = torch.vstack(pseudo_feats)
                        pseudo_targets = torch.tensor(pseudo_targets).to(self.device)
                        output = self.model.linear(pseudo_feats)
                        pseudo_loss = torch.nn.functional.cross_entropy(output, pseudo_targets)
                        horde_loss += pseudo_loss

                # Normalize and add to the main training loss
                loss += horde_loss / targets.size(0)
                val_loss += loss.item() * targets.size(0)
                val_hits += torch.sum(torch.argmax(outputs, dim=1) == targets.to(self.device)).cpu().item()
                val_num_elements += targets.size(0)
        return val_loss / val_num_elements, val_hits / val_num_elements

    def contrastive_loss(self, outputs, targets):
        # Get image pairs
        try:
            positive_pairs, negative_pairs = self._pair_selector.get_pairs(outputs, targets)
        except IndexError:
            return torch.tensor(0)
        if positive_pairs == [] and negative_pairs == []:
            return torch.tensor(0)
        if outputs.is_cuda:
            positive_pairs = positive_pairs.to(outputs.device)
            negative_pairs = negative_pairs.to(outputs.device)
        # Make sure that the sets are not empty
        if positive_pairs.nelement() == 0:
            return torch.tensor(0.0, device=outputs.device)
        # Apply the contrastive loss with margin
        positive_loss = (outputs[positive_pairs[:, 0]] - outputs[positive_pairs[:, 1]]).pow(2).sum(1)
        dist = (outputs[negative_pairs[:, 0]] - outputs[negative_pairs[:, 1]]).pow(2).sum(1).sqrt()
        negative_loss = torch.nn.functional.relu(self.ml_margin - dist).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()

    def _train_feature_extractor(self, t, fe_idx: int, trn_loader: DataLoader, val_loader: DataLoader) -> torch.nn.Module:
        """Trains a new feature Extractor """
        def cross_entropy_lwf(outputs, targets, exp=1.0, size_average=True, eps=1e-5):
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

        use_lwf_for_training = False
        if not self._initial_training_done:
            model = self.model.get_initial_model()
        else:
            if self.model.has_room_to_grow() or fe_idx == -1:
                model = self.model.get_fe_model()
            else:
                base_fe = self.model.feature_extractors[fe_idx]
                model = copy.deepcopy(base_fe)
                for p in model.parameters():
                    p.requires_grad = True
                base_fe.eval()

                if self.use_lwf_fe:
                    use_lwf_for_training = True
                    # This  head should simulate the predictions of a single fe
                    pseudo_head_lwf = Linear(self.single_fe_latent_dim, len(self._known_classes))

                    start_dim = fe_idx * self.single_fe_latent_dim
                    end_dim = (fe_idx + 1) * self.single_fe_latent_dim
                    pseudo_head_lwf.requires_grad_(False)
                    pseudo_head_lwf.weight[:, :] = self.model.linear.weight[:, start_dim:end_dim]
                    pseudo_head_lwf.bias[:] = self.model.linear.bias[:]

                    pseudo_head_lwf.eval()
                    pseudo_head_lwf.to(self.device)

        # Create a head for CE and a head for Metric Learning
        n_classes_this_exp = len(self._cur_val_to_idx_dict.keys())
        if self.use_self_supervision:
            ce_head = Linear(self.single_fe_latent_dim if self._initial_training_done else self.model.init_out_size, n_classes_this_exp * 4)
        else:
            ce_head = Linear(self.single_fe_latent_dim if self._initial_training_done else self.model.init_out_size, n_classes_this_exp)

        ml_head = Linear(self.single_fe_latent_dim if self._initial_training_done else self.model.init_out_size, self.ml_dims)
        model.to(self.device)
        ce_head.to(self.device)
        ml_head.to(self.device)

        class_lookup_id = {}
        for i, k in enumerate(self._cur_val_to_idx_dict.keys()):
            class_lookup_id[k] = i

        # Warmup - period for the CE Head
        if not self.model.has_room_to_grow():
            warmup_optimizer = torch.optim.Adam(ce_head.parameters(), lr=0.001, weight_decay=self.wd)
            # Loop epochs -- train warm-up head
            for e in range(self.warmup_epochs):
                warmup_hits = 0.0
                warmup_loss = 0.0
                for images, targets in trn_loader:
                    images = images.to(self.device)
                    targets = targets.cpu().apply_(lambda val: class_lookup_id[val]).to(self.device)

                    if self.use_self_supervision:
                        orig_channels, orig_height, orig_width = images.size(1), images.size(2), images.size(3)
                        # self-supervised learning based label augmentation
                        images = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1)
                        images = images.view(-1, orig_channels, orig_height, orig_width)
                        targets = torch.stack([targets * 4 + k for k in range(4)], 1).view(-1)

                    feats = model(images)
                    output = ce_head(feats)

                    loss = torch.nn.functional.cross_entropy(output, targets)
                    warmup_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(ce_head.parameters(), self.clipgrad)
                    warmup_optimizer.step()

                    warmup_loss += loss.cpu().detach().item()
                    pred = torch.argmax(output, dim=1)
                    warmup_hits += torch.sum(pred == targets).cpu().detach().item()

                print('FE Training {}| Warm-up Epoch {:3d} | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(t, e + 1, warmup_loss / len(trn_loader), 100 * warmup_hits / len(trn_loader.dataset)))

        model.train()
        # LR will be overwritten later -- add params from main_model, CE head and ML head
        params = list(model.parameters())
        if self.training_method_fe in ["ce", "ce_ml"]:
            params += list(ce_head.parameters())

        if self.training_method_fe in ["ml", "ce_ml"]:
            params += list(ml_head.parameters())

        optimizer = Adam(params, lr=0.001, weight_decay=self.wd)
        if self.use_adaptive_alpha:
            wk_alpha = 0.001
        else:
            wk_alpha = self.alpha

        # Starting best model is current model
        total_epochs = 0
        best_loss = np.inf
        best_model = None
        best_ce_head = None
        best_ml_head = None

        # Training loop
        for schedule in zip(self.fe_lr, self.fe_epochs):
            current_lr = schedule[0]
            patience = self.lr_patience
            _set_learning_rate(optimizer, current_lr)
            for epoch in range(schedule[1]):
                total_epochs += 1
                metric_ce_loss, metric_ml_loss, metric_lwf_loss = 0.0, 0.0, 0.0
                hits, total_elements = 0, 0
                start_time = time.time()
                model.train(), ce_head.train(), ml_head.train()
                for x, y in trn_loader:
                    x_device = x.to(self.device)
                    y_device = y.cpu().apply_(lambda val: class_lookup_id[val]).to(self.device)

                    if self.use_self_supervision:
                        orig_channels, orig_height, orig_width = x_device.size(1), x_device.size(2), x_device.size(3)
                        # self-supervised learning based label augmentation
                        x_device = torch.stack([torch.rot90(x_device, k, (2, 3)) for k in range(4)], 1)
                        x_device = x_device.view(-1, orig_channels, orig_height, orig_width)
                        y_device = torch.stack([y_device * 4 + k for k in range(4)], 1).view(-1)

                    feats = model(x_device)
                    losses = torch.tensor(0.0, device=self.device)
                    # Forward through the CE-head
                    if self.training_method_fe in ["ce", "ce_ml"]:
                        ce_out = ce_head(feats)
                        ce_loss = torch.nn.functional.cross_entropy(ce_out, y_device)
                        metric_ce_loss += ce_loss.detach().item() * y_device.size(0)
                        # Apply working alpha
                        if torch.isnan(ce_loss):
                            warnings.warn(f"FE Training {t} | Epoch {total_epochs} | CE Loss is NaN")
                            continue
                        else:
                            losses = (1 - wk_alpha) * ce_loss
                        hits += torch.sum(ce_out.argmax(dim=1) == y_device).detach().item()

                    if wk_alpha > 0.0 and self.training_method_fe in ["ml", "ce_ml"]:
                        # Forward through the ML-head
                        if self.use_self_supervision:
                            ml_feats = feats
                            ml_targets = y_device // 4  # class index mapping already in place with apply_
                        else:
                            ml_feats = feats
                            ml_targets = y_device

                        ml_out = ml_head(ml_feats)
                        ml_loss = self.contrastive_loss(ml_out, ml_targets)
                        if torch.isnan(ml_loss):
                            warnings.warn(f"FE Training {t} | Epoch {total_epochs} | ML Loss has become NAN skipping batch ... ")
                        else:
                            losses += wk_alpha * ml_loss
                            metric_ml_loss += ml_loss.detach().item() * y.size(0)

                    if use_lwf_for_training:
                        with torch.no_grad():
                            outputs_old_model = pseudo_head_lwf(base_fe(x_device))
                        lwf_loss = self.fe_lwf_lamb * cross_entropy_lwf(ce_out, outputs_old_model, 1.0 / self.fe_lwf_T)
                        if torch.isnan(lwf_loss):
                            warnings.warn(f"FE Training {t} | Epoch {total_epochs} | LWF Loss has become NAN skipping batch ... ")
                        else:
                            losses += lwf_loss
                            metric_lwf_loss += lwf_loss.detach().item() * y_device.size(0)

                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()
                    total_elements += y_device.size(0)

                    if torch.isnan(losses):
                        print("LOSS IS NAN skipping / aborting model training")
                        if best_model is None:
                            return self.model.get_initial_model()
                        else:
                            return best_model


                end_time = time.time()
                if total_elements != 0:
                    metric_ml_loss /= total_elements
                    metric_ce_loss /= total_elements
                    metric_lwf_loss /= total_elements
                else:
                    metric_ml_loss = np.inf
                    metric_ce_loss = np.inf
                    metric_lwf_loss = np.inf
                    total_elements = 10000000


                print(f"FE Training {t} | Epoch {total_epochs} "
                      f"| Training running: loss={metric_ce_loss:.3f}, ml_loss={metric_ml_loss:.3f}, lwf_loss={metric_lwf_loss:.3f}"
                      f", acc={hits / total_elements * 100.0:6.2f}%, alpha={wk_alpha:.4f} | time={end_time - start_time:.4f}s |", end="")

                # Keep track of the model with the best (lowest) loss
                if len(val_loader) > 0:
                    val_ce_loss, val_ml_loss = 0.0, 0.0
                    val_hits, val_elements = 0, 0
                    start_time = time.time()
                    with torch.inference_mode():
                        model.eval(), ce_head.eval(), ml_head.eval()
                        for x, y in val_loader:
                            x_device = x.to(self.device)
                            y_device = y.cpu().apply_(lambda val: class_lookup_id[val]).to(self.device)

                            if self.use_self_supervision:
                                orig_channels, orig_height, orig_width = x_device.size(1), x_device.size(2), x_device.size(3)
                                # self-supervised learning based label augmentation
                                x_device = torch.stack([torch.rot90(x_device, k, (2, 3)) for k in range(4)], 1)
                                x_device = x_device.view(-1, orig_channels, orig_height, orig_width)
                                y_device = torch.stack([y_device * 4 + k for k in range(4)], 1).view(-1)

                            feats = model(x_device)
                            # Forward through the CE-head
                            if self.training_method_fe in ["ce", "ce_ml"]:
                                ce_out = ce_head(feats)
                                ce_loss = torch.nn.functional.cross_entropy(ce_out, y_device)
                                val_ce_loss += ce_loss.detach().item() * y.size(0)
                                val_hits += torch.sum(ce_out.argmax(dim=1) == y_device).detach().item()
                            if wk_alpha > 0.0 and self.training_method_fe in ["ml", "ce_ml"]:
                                # Forward through the ML-head
                                if self.use_self_supervision:
                                    ml_feats = feats
                                    ml_targets = y_device // 4  # class index mapping already in place with apply_
                                else:
                                    ml_feats = feats
                                    ml_targets = y_device

                                ml_out = ml_head(ml_feats)
                                ml_loss = self.contrastive_loss(ml_out, ml_targets)
                                if torch.isnan(ml_loss):
                                    warnings.warn(f"FE Training {t} | Epoch {total_epochs} | ML Loss has become NAN skipping batch ... ")
                                else:
                                    val_ml_loss += ml_loss.detach().item() * y.size(0)
                            val_elements += y_device.size(0)
                    end_time = time.time()
                    if val_elements > 0:
                        val_ce_loss /= val_elements
                        val_ml_loss /= val_elements
                    else:
                        val_ce_loss = np.inf
                        val_ml_loss = np.inf
                        val_elements = 100000000
                    print(f" Valid loss={val_ce_loss :.3f}, ml_loss={val_ml_loss:.3f}, acc={val_hits / val_elements * 100.0:.2f}% | time={end_time - start_time:.4f}s |", end="")

                    if self.use_early_stopping:
                        total_val_loss = (1 - wk_alpha) * val_ce_loss + wk_alpha * val_ml_loss if not self.use_adaptive_alpha else val_ce_loss + val_ml_loss
                        if total_val_loss < best_loss:
                            best_loss = total_val_loss
                            best_model = model.state_dict()
                            best_ce_head = ce_head.state_dict()
                            best_ml_head = ml_head.state_dict()
                            patience = self.lr_patience
                            print('*', end='')
                        else:
                            # if the loss does not go down, decrease patience
                            patience -= 1
                            if patience <= 0:
                                # if it runs out of patience, reduce the learning rate
                                current_lr /= self.lr_factor
                                print(' lr={:.1e}'.format(current_lr), end='')
                                if current_lr < self.lr_min:
                                    # if the lr decreases below minimum, stop the training session for current schedule
                                    print()
                                    model.load_state_dict(best_model)
                                    ce_head.load_state_dict(best_ce_head)
                                    ml_head.load_state_dict(best_ml_head)
                                    break
                                # reset patience and recover best model so far to continue training
                                patience = self.lr_patience
                                _set_learning_rate(optimizer, current_lr)
                                model.load_state_dict(best_model)
                                ce_head.load_state_dict(best_ce_head)
                                ml_head.load_state_dict(best_ml_head)
                print()

                # Adaptive alpha: balance the working alpha to accommodate the loss difference in magnitude
                if self.use_adaptive_alpha:
                    wk_alpha = metric_ml_loss / (metric_ml_loss + metric_ce_loss + 1e-12)
                    if math.isnan(wk_alpha):
                        warnings.warn("Working alpha collapsed to NaN, setting back to 0.5")
                        wk_alpha = 0.5

        # Return the trained feature extractor
        return model

    def _calculate_embedding_statistics(self, trn_loader: DataLoader):
        with torch.inference_mode():
            for cls_idx in self._cur_val_to_idx_dict:
                cls_sampler = SubsetRandomSampler(indices=self._cur_val_to_idx_dict[cls_idx])
                cls_loader = DataLoader(trn_loader.dataset, batch_size=trn_loader.batch_size, shuffle=False, sampler=cls_sampler)
                fe_embed_dict = {}
                initial_fe_embed_dict = None
                for i in range(self.__num_iterations_for_mean):
                    for image, target in cls_loader:
                        image = image.to(self.device)

                        for num_fe, feature_extractor_i in enumerate(self.model.feature_extractors):
                            # Skip embedding calculation when better estimate was already calculated
                            if self.max_samples_seen[cls_idx, num_fe] >= len(self._cur_val_to_idx_dict[cls_idx]) and not self.acc_prototype:
                                continue

                            feature_extractor_i.eval()
                            features = feature_extractor_i(image)
                            if num_fe not in fe_embed_dict:
                                fe_embed_dict[num_fe] = features
                            else:
                                fe_embed_dict[num_fe] = torch.vstack([fe_embed_dict[num_fe], features])

                        if self.initial_max_samples_seen[cls_idx] < len(self._cur_val_to_idx_dict):
                            self.model.initial_feature_extractor.eval()
                            features = self.model.initial_feature_extractor(image)
                            initial_fe_embed_dict = features if initial_fe_embed_dict is None else torch.vstack([initial_fe_embed_dict, features])

                for num_fe in fe_embed_dict.keys():
                    if self.acc_prototype:
                        total_samples = self.max_samples_seen[cls_idx, num_fe] + len(self._cur_val_to_idx_dict[cls_idx])
                        self.mean_embedding[cls_idx, num_fe, :] = torch.mean(fe_embed_dict[num_fe], dim=0) * len(self._cur_val_to_idx_dict[cls_idx]) / total_samples + self.mean_embedding[cls_idx, num_fe,:] * self.max_samples_seen[cls_idx, num_fe] / total_samples
                        self.std_embedding[cls_idx, num_fe, :] = torch.std(fe_embed_dict[num_fe], dim=0) * len(self._cur_val_to_idx_dict[cls_idx]) / total_samples + self.std_embedding[cls_idx, num_fe, :] * self.max_samples_seen[ cls_idx, num_fe] / total_samples
                        self.max_samples_seen[cls_idx, num_fe] = len(self._cur_val_to_idx_dict[cls_idx])
                    else:
                        self.mean_embedding[cls_idx, num_fe, :] = torch.mean(fe_embed_dict[num_fe], dim=0)
                        self.std_embedding[cls_idx, num_fe, :] = torch.std(fe_embed_dict[num_fe], dim=0)
                        self.max_samples_seen[cls_idx, num_fe] = len(self._cur_val_to_idx_dict[cls_idx])

                if initial_fe_embed_dict is not None:
                    if self.acc_prototype:
                        total_samples = self.initial_max_samples_seen[cls_idx] + len(self._cur_val_to_idx_dict[cls_idx])
                        self.initial_mean_embedding[cls_idx, :] = torch.mean(initial_fe_embed_dict, dim=0) * len(self._cur_val_to_idx_dict[cls_idx]) / total_samples + self.initial_mean_embedding[cls_idx, :] * self.initial_max_samples_seen[cls_idx] / total_samples
                        self.initial_std_embedding[cls_idx, :] = torch.std(initial_fe_embed_dict, dim=0) * len(self._cur_val_to_idx_dict[cls_idx]) / total_samples + self.initial_std_embedding[cls_idx, :] * self.initial_max_samples_seen[cls_idx] / total_samples
                        self.initial_max_samples_seen[cls_idx] = len(self._cur_val_to_idx_dict[cls_idx])
                    else:
                        self.initial_mean_embedding[cls_idx, :] = torch.mean(initial_fe_embed_dict, dim=0)
                        self.initial_std_embedding[cls_idx, :] = torch.std(initial_fe_embed_dict, dim=0)
                        self.initial_max_samples_seen[cls_idx] = len(self._cur_val_to_idx_dict[cls_idx])

    def calculate_metrics(self, outputs, targets):
        """Contains the main Task-Agnostic metrics"""
        pred = outputs.argmax(dim=1)
        hits = (pred == targets.to(self.device)).float()
        return hits

    def _calculate_accuracy_score(self, trn_loader: DataLoader):
        with torch.inference_mode():
            self.model.eval()
            hits = 0
            total_elements = 0
            for x, y in trn_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                total_elements += y.shape[0]
                pred = self.model(x)
                hits += (total_elements == pred.argmax(dim=1)).sum().item()

        self.model.train()
        return hits / total_elements

    def _decide_if_fe_should_be_added(self, t, trn_loader, fe_idx, feature_extractor):
        """This point introduces the option to modify whether the """
        if not self._initial_training_done:
            return True, -1

        if self.fe_selection == "distribution_overlap":
            def calculate_internal_prototype_overlap(prototypes):
                score = 0.0
                for i, p in enumerate(prototypes):
                    for p_other in prototypes[i+1:]:
                        common_classes = p.get_common_class_list(p_other)
                        if not common_classes:
                            # I think this is impossible but lets see
                            print("No common classes assume complete overlap!")
                            score += 1
                            continue

                        p_sub = p.create_sub_class_prototype(common_classes)
                        p_other_sub = p_other.create_sub_class_prototype(common_classes)
                        score += p_sub.calculate_distribution_overlap(p_other_sub)
                return score

            # Update Class statistics
            self._calculate_embedding_statistics(trn_loader)
            current_prototypes = convert_class_prototypes(self.initial_mean_embedding, self.initial_std_embedding,
                                                          self.mean_embedding, self.std_embedding, self.initial_max_samples_seen,
                                                          self.max_samples_seen)
            new_trained_prototype = calculate_partial_class_prototype(trn_loader, feature_extractor, self.device, self._cur_val_to_idx_dict)

            best_prototype_score = calculate_internal_prototype_overlap(current_prototypes)
            replace_fe = False
            fe_idx = -1
            print(f"Current Score for distribution overlap: {best_prototype_score}")

            # Find best configuration for minimal internal class overlap!
            for idx in range(self.num_fe + 1) if not self.model.has_room_to_grow else [len(self.model.feature_extractors)]:
                fe_selection_prototypes = []
                for p in current_prototypes:
                    fe_config_selection = copy.deepcopy(p.fe_idx_list)
                    try:
                        fe_config_selection.remove(idx+1)
                    except ValueError:
                        pass  # If replacing element not in current list just skip
                    fe_p = p.create_sub_class_prototype(fe_config_selection)
                    if fe_p.class_id in new_trained_prototype:
                        fe_p.extend_class_prototype(new_trained_prototype[fe_p.class_id]["mean"],
                                                    new_trained_prototype[fe_p.class_id]["std"], self.num_fe+1)
                    fe_selection_prototypes.append(fe_p)
                fe_selection_prototype_score = calculate_internal_prototype_overlap(fe_selection_prototypes)
                print(f"Score for position {idx}: {fe_selection_prototype_score}")
                if fe_selection_prototype_score < best_prototype_score:
                    best_prototype_score = fe_selection_prototype_score
                    fe_idx = idx
                    replace_fe = True

            print(f"Replace Feature extractor on position {fe_idx}: {replace_fe}")
            return replace_fe, fe_idx
        else: # for all other methods we keep to the orginal plan :)
            return True, fe_idx

