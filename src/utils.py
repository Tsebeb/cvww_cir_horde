import os
from contextlib import contextmanager

import torch
import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, Subset, ConcatDataset, DataLoader
from torchmetrics.classification import MulticlassConfusionMatrix

from datasets.base_dataset import BaseDataset
from datasets.memory_dataset import MemoryDataset
from networks.cil_network import CIL_Net

cudnn_deterministic = True

def _get_dataset_transform(dataset: Dataset):
    if isinstance(dataset, ConcatDataset):
        return _get_dataset_transform(dataset.datasets[0])
    elif isinstance(dataset, Subset):
        return _get_dataset_transform(dataset.dataset)
    else:
        return dataset.transform


def _get_all_targets(dataset: Dataset):
    if isinstance(dataset, ConcatDataset):
        partial_targets = []
        for ds in dataset.datasets:
            partial_targets.append(_get_all_targets(ds))
        return torch.cat(partial_targets)
    elif isinstance(dataset, Subset):
        targets = _get_all_targets(dataset.dataset)
        return targets[dataset.indices]
    else:
        return dataset.labels


def _get_unique_targets(dataset: Dataset):
    all_targets = _get_all_targets(dataset)
    return torch.unique(all_targets)


def get_transform_from_dataloader(dl: DataLoader):
    return _get_dataset_transform(dl.dataset)


@contextmanager
def override_transform_dataloader(dataloader: DataLoader, transform, shuffle):
    def _set_dataset_transform(dataset: Dataset, new_transform):
        if isinstance(dataset, ConcatDataset):
            for ds in dataset.datasets:
                _set_dataset_transform(ds, new_transform)
        elif isinstance(dataset, Subset):
            _set_dataset_transform(dataset.dataset, new_transform)
        else:
            dataset.transform = new_transform

    try:
        dataset = dataloader.dataset
        orig_transform = _get_dataset_transform(dataset)
        _set_dataset_transform(dataset, transform)
        new_dataloader = DataLoader(dataset,
                                    batch_size=dataloader.batch_size,
                                    shuffle=shuffle,
                                    num_workers=dataloader.num_workers,
                                    pin_memory=dataloader.pin_memory)
        yield new_dataloader
    finally:
        _set_dataset_transform(dataloader.dataset, orig_transform)

def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    #torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.enabled = True


def print_summary(acc_up2now, acc, forg):
    """Print summary of results"""
    for name, metric in zip(['Acc Up2Now', 'Acc', 'Forg'], [acc_up2now, acc, forg]):
        print('*' * 108)
        print(name)
        for i in range(metric.shape[0]):
            print('{:5.1f}% '.format(100 * metric[i]), end='')
        print('\tAvg.:{:5.1f}% '.format(100 * metric.mean()), end='')
        print()
    print('*' * 108)

def get_features_embedd_for_modell(model: CIL_Net, dataloader, setup_model_eval=True):
    if setup_model_eval:
        model.eval()
    else:
        model.train()

    model_device = "cpu"
    for p in model.parameters():
        model_device = p.device
        break

    with torch.no_grad():
        total_features = []
        for x, _ in dataloader:
            x = x.to(model_device)
            _, features = model(x, return_features=True)
            total_features.append(features.cpu())
    return torch.vstack(total_features)


def get_targets(dataset: Dataset):

    if isinstance(dataset, Subset):
        all_targets = get_targets(dataset.dataset)
        return all_targets[dataset.indices]
    elif isinstance(dataset, ConcatDataset):
        partial_targets = []
        for ds in dataset.datasets:
            partial_targets.append(get_targets(ds))
        return torch.cat(partial_targets)
    elif isinstance(dataset, MemoryDataset) or isinstance(dataset, BaseDataset):
        return dataset.labels
    else:
        raise RuntimeError("Unsupported Dataset BaseType Implement rule here")


def _create_copy_dataloader(base_dataloader: DataLoader, indices, shuffle=False):
    base_ds = base_dataloader.dataset
    new_ds = Subset(base_ds, indices)
    return DataLoader(new_ds, batch_size=base_dataloader.batch_size, num_workers=base_dataloader.num_workers,
                      pin_memory=base_dataloader.pin_memory, shuffle=shuffle)


def get_class_seperated_embeddings(model, dataloader, samples_per_class=-1):
    targets = get_targets(dataloader.dataset)
    classes = torch.unique(targets)
    embeddings_per_class = {}
    # Get present classes
    for c in classes:
        subset_targets_dl = _create_copy_dataloader(dataloader, targets[targets == c])
        class_embeddings = get_features_embedd_for_modell(model, subset_targets_dl)
        if samples_per_class > 0:
            class_embeddings = class_embeddings[:samples_per_class]

        embeddings_per_class[c.item()] = class_embeddings

    return embeddings_per_class


def create_embeddings_plot(model: CIL_Net, dataloader: DataLoader, class_colors, trained_projection, samples_per_class=-1, exemplars=None):
    """
    This method plots the current
    """
    embeddings_ds = get_class_seperated_embeddings(model, dataloader, samples_per_class)
    exemplars_embeddings = get_class_seperated_embeddings(model, exemplars, samples_per_class) if exemplars is not None else None

    # Convert embeddings
    for key in embeddings_ds:
        embeddings_ds[key] = trained_projection.transform(embeddings_ds[key])

    if exemplars_embeddings is not None:
        for key in exemplars_embeddings:
            exemplars_embeddings[key] = trained_projection.transform(exemplars_embeddings[key])

    plt.title("Embeddings ")
    for key in embeddings_ds:
        plt.scatter(embeddings_ds[key][:, 0], embeddings_ds[key][:, 1], cmap=class_colors[key], alpha=0.5)

    if exemplars_embeddings is not None:
        for key in embeddings_ds:
            plt.scatter(embeddings_ds[key][:, 0], embeddings_ds[key][:, 1], cmap=class_colors[key], marker="^")

    plt.show()


def get_confusion_matrix(model: torch.nn.Module, loader: DataLoader, num_classes: int, device: str):
    with torch.inference_mode():
        model.eval()
        model.to(device)
        m_conf = MulticlassConfusionMatrix(num_classes)
        for x, y in loader:
            x = x.to(device)
            pred = model(x).argmax(dim=1)
            m_conf.update(pred.cpu(), y)
    conf_mat = m_conf.compute()
    return conf_mat


def print_confusion_matrix(conf_matrix: torch.Tensor):
    assert len(conf_matrix.shape) == 2, "Check if matrix has a correct form"
    print("-" * 80)
    for r_idx in range(conf_matrix.shape[0]):
        for c_idx in range(conf_matrix.shape[1]):
            print(f"{conf_matrix[r_idx, c_idx].long():>5}\t", end="")
        print()
    print("-" * 80)
