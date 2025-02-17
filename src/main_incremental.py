import os
import sys
import time
import warnings

import torch
import argparse
import importlib
import numpy as np
from functools import reduce

import utils
import approach
from loggers.exp_logger import MultiLogger
from datasets.dataset_config import dataset_config
from last_layer_analysis import last_layer_analysis
from networks import allmodels


def main(argv=None):
    """ Changes from FACIL include:
        - removal of warm-up and multi-softmax options
        - removal of number classes first task
        - reduction of setting to single head -- future: grow the head as new classes are seen
        - management of tasks and classes by scenario
    """
    tstart = time.time()
    # Arguments
    parser = argparse.ArgumentParser(description='FACIL - Framework for Analysis of Class Incremental Learning')

    # miscellaneous args
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU (default=%(default)s)')
    parser.add_argument('--results-path', type=str, default='../results',
                        help='Results path (default=%(default)s)')
    parser.add_argument('--exp-name', default=None, type=str,
                        help='Experiment name (default=%(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default=%(default)s)')
    parser.add_argument('--log', default=['disk'], type=str, choices=['disk', 'tensorboard'],
                        help='Loggers used (disk, tensorboard) (default=%(default)s)', nargs='*', metavar="LOGGER")
    parser.add_argument('--save-models', action='store_true',
                        help='Save trained models (default=%(default)s)')
    parser.add_argument('--last-layer-analysis', action='store_true',
                        help='Plot last layer analysis (default=%(default)s)')
    parser.add_argument('--no-cudnn-deterministic', action='store_true',
                        help='Disable CUDNN deterministic (default=%(default)s)')
    # dataset args
    parser.add_argument('--datasets', default=['mnist'], type=str, choices=list(dataset_config.keys()),
                        help='Dataset or datasets used (default=%(default)s)', nargs='+', metavar="DATASET")
    parser.add_argument('--validation', default=0.0, type=float, help='Fraction of validation')
    parser.add_argument('--num-workers', default=4, type=int, required=False,
                        help='Number of subprocesses to use for dataloader (default=%(default)s)')
    parser.add_argument('--pin-memory', action="store_true", required=False,
                        help='Copy Tensors into CUDA pinned memory before returning them (default=%(default)s)')
    parser.add_argument('--batch-size', default=64, type=int, required=False,
                        help='Number of samples per batch to load (default=%(default)s)')
    parser.add_argument('--num-tasks', default=10, type=int, required=False,
                        help='Number of tasks per dataset (default=%(default)s)')
    parser.add_argument('--use-valid-only', action='store_true',
                        help='Use validation split instead of test (default=%(default)s)')
    parser.add_argument('--stop-at-task', default=0, type=int, required=False,
                        help='Stop training after specified task (default=%(default)s)')
    # model args
    parser.add_argument('--network', default='LeNet', type=str, choices=allmodels,
                        help='Network architecture used (default=%(default)s)', metavar="NETWORK")
    parser.add_argument('--keep-existing-head', action='store_true',
                        help='Disable removing classifier last layer (default=%(default)s)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained backbone (default=%(default)s)')
    # training args
    parser.add_argument('--approach', default='finetuning', type=str, choices=approach.__all__,
                        help='Learning approach used (default=%(default)s)', metavar="APPROACH")
    parser.add_argument('--nepochs', default=[200], type=int, required=False, nargs="+",
                        help='Number of epochs per training session (default=%(default)s)')
    parser.add_argument('--lr', default=[0.001], type=float, required=False, nargs="+",
                        help='Starting learning rate (default=%(default)s)')
    parser.add_argument('--lr-min', default=1e-5, type=float, required=False,
                        help='Minimum learning rate (default=%(default)s)')
    parser.add_argument('--lr-factor', default=10, type=float, required=False,
                        help='Learning rate decreasing factor (default=%(default)s)')
    parser.add_argument("--dont-use-early-stopping", action="store_true", help="")
    parser.add_argument('--lr-patience', default=5, type=int, required=False,
                        help='Maximum patience to wait before decreasing learning rate (default=%(default)s)')
    parser.add_argument('--clipping', default=10000, type=float, required=False,
                        help='Clip gradient norm (default=%(default)s)')
    parser.add_argument('--momentum', default=0.9, type=float, required=False,
                        help='Momentum factor (default=%(default)s)')
    parser.add_argument('--weight-decay', default=0.0, type=float, required=False,
                        help='Weight decay (L2 penalty) (default=%(default)s)')
    parser.add_argument('--fix-bn', action='store_true',
                        help='Fix batch normalization after first task (default=%(default)s)')
    parser.add_argument('--eval-on-train', action='store_true',
                        help='Show train loss and accuracy (default=%(default)s)')
    parser.add_argument("--loader-method", default="random_repetition", help="The method used for "
                                                                             "generating the repetition scenario used.")

    # Args -- Incremental Learning Framework
    args, extra_args = parser.parse_known_args(argv)
    args.results_path = os.path.expanduser(args.results_path)
    assert 0.0 <= args.validation < 1.0, "validation fraction needs to be between [0, 1)"
    if len(args.lr) > 1 and args.validation > 0.0 and not args.dont_use_early_stopping:
        print("WARNING multiple step learning rates with dynamic early stopping not supported")
        sys.exit(-1)

    use_early_stopping = not args.dont_use_early_stopping
    if args.validation == 0.0 and not args.dont_use_early_stopping:
        warnings.warn("WARNING: Early Stopping will be disabled since validation split is 0.0")
        use_early_stopping = False

    base_kwargs = dict(nepochs=args.nepochs, lr=args.lr, lr_min=args.lr_min, lr_factor=args.lr_factor,
                       lr_patience=args.lr_patience, clipgrad=args.clipping, momentum=args.momentum,
                       wd=args.weight_decay, fix_bn=args.fix_bn, eval_on_train=args.eval_on_train,
                       use_early_stopping=use_early_stopping)

    if args.no_cudnn_deterministic:
        print('WARNING: CUDNN Deterministic will be disabled.')
        utils.cudnn_deterministic = False

    utils.seed_everything(seed=args.seed)
    print('=' * 108)
    print('Arguments =')
    for arg in np.sort(list(vars(args).keys())):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 108)

    # Args -- CUDA
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = 'cuda'
    else:
        print('WARNING: [CUDA unavailable] Using CPU instead!')
        device = 'cpu'
    # Multiple gpus
    # if torch.cuda.device_count() > 1:
    #     self.C = torch.nn.DataParallel(C)
    #     self.C.to(self.device)
    ####################################################################################################################

    # Args -- Continual Learning Approach
    from approach.incremental_learning import Inc_Learning_Appr
    Appr = getattr(importlib.import_module(name='approach.' + args.approach), 'Appr')
    assert issubclass(Appr, Inc_Learning_Appr)
    appr_args, extra_args = Appr.extra_parser(extra_args)
    print('Approach arguments =')
    for arg in np.sort(list(vars(appr_args).keys())):
        print('\t' + arg + ':', getattr(appr_args, arg))
    print('=' * 108)

    # Args -- Exemplars Management
    from datasets.exemplars_dataset import ExemplarsDataset
    Appr_ExemplarsDataset = Appr.exemplars_dataset_class()
    if Appr_ExemplarsDataset:
        assert issubclass(Appr_ExemplarsDataset, ExemplarsDataset)
        appr_exemplars_dataset_args, extra_args = Appr_ExemplarsDataset.extra_parser(extra_args)
        print('Exemplars dataset arguments =')
        for arg in np.sort(list(vars(appr_exemplars_dataset_args).keys())):
            print('\t' + arg + ':', getattr(appr_exemplars_dataset_args, arg))
        print('=' * 108)
    else:
        appr_exemplars_dataset_args = argparse.Namespace()

    # Args -- Scenario Config
    from scenarios.manage_loaders import Manage_Loaders
    LoaderClass = getattr(importlib.import_module(name="scenarios." + args.loader_method), "Scenario_Loader")
    assert issubclass(LoaderClass, Manage_Loaders)
    loader_args, extra_args = LoaderClass.extra_parser(extra_args)
    print('Scenario Loader arguments =')
    for arg in np.sort(list(vars(loader_args).keys())):
        print('\t' + arg + ':', getattr(loader_args, arg))
    print('=' * 108)

    # Args -- Network Wrapper
    from networks.cil_network import CIL_Net
    ModelClass = Appr.get_model_class()
    assert issubclass(ModelClass, CIL_Net)
    model_args, extra_args = ModelClass.extra_parser(extra_args)
    print('Model arguments =')
    for arg in np.sort(list(vars(model_args).keys())):
        print('\t' + arg + ':', getattr(model_args, arg))
    print('=' * 108)

    assert len(extra_args) == 0, "Unused args: {}".format(' '.join(extra_args))
    ####################################################################################################################

    # Log all arguments
    full_exp_name = reduce((lambda x, y: x[0] + y[0]), args.datasets) if len(args.datasets) > 0 else args.datasets[0]
    full_exp_name += '_' + args.approach
    if args.exp_name is not None:
        full_exp_name += '_' + args.exp_name
    logger = MultiLogger(args.results_path, full_exp_name, loggers=args.log, save_models=args.save_models)
    logger.log_args(argparse.Namespace(**args.__dict__, **appr_args.__dict__, **appr_exemplars_dataset_args.__dict__,
                                       **loader_args.__dict__, **model_args.__dict__))

    # Loaders
    utils.seed_everything(seed=args.seed)
    loaders = LoaderClass(args.datasets, args.validation, args.num_tasks, args.batch_size, args.num_workers,
                          args.pin_memory, **loader_args.__dict__)
    # Split loaders and apply class ordering!
    loaders.split_loaders()
    max_task = args.stop_at_task if args.stop_at_task > 0 else args.num_tasks

    # Network and Approach instances
    utils.seed_everything(seed=args.seed)
    net = ModelClass(args.network, args.pretrained, not args.keep_existing_head, **model_args.__dict__)
    utils.seed_everything(seed=args.seed)

    # taking transformations and class indices from first train dataset
    first_train_ds = loaders.total_trn_loader.dataset
    transform, class_indices = first_train_ds.transform, first_train_ds.class_indices
    appr_kwargs = {**base_kwargs, **dict(logger=logger, **appr_args.__dict__)}
    if Appr_ExemplarsDataset:
        appr_kwargs['exemplars_dataset'] = Appr_ExemplarsDataset(transform, class_indices, **appr_exemplars_dataset_args.__dict__)
    utils.seed_everything(seed=args.seed)
    appr = Appr(net, device, **appr_kwargs)

    # Loop tasks
    accuracies_task_total = np.zeros((max_task, ))
    forgetting_task_total = np.zeros((max_task, ))
    accuracies_task = np.zeros((max_task, ))
    forgetting_task = np.zeros((max_task, ))
    acc_triangle = np.zeros((max_task, max_task))

    print("Class Mapping (original class -> ordered) for this run ", loaders.class_mapping)
    print("Inverse Class Mapping (ordered -> original class) for this run ", loaders.inverse_class_mapping)
    for t in range(loaders.num_tasks):
        # Early stop tasks if flag
        if t >= max_task:
            continue
        # modify head to fit the current number of classes
        net.modify_head(len(loaders.get_classes_present_so_far(t)))
        net.to(device)
        print('*' * 108)
        print('Task {:2d}'.format(t))
        print("Classes present in this Task: ", loaders.get_classes_present_in_t(t))
        print('*' * 108)

        # Train
        appr.train(t, loaders.get_trn_loader(t), loaders.get_val_loader(t) if args.approach != "joint" else loaders.get_accumulated_val_loader(t))
        print('-' * 108)

        # Test - Task agnostic only !
        test_loader = loaders.get_tst_loader(t) if not args.use_valid_only else loaders.get_val_loader(t)
        print(f"Number of Test samples for evaluation {len(test_loader.dataset)}")
        total_test_loss, accuracies_task_total[t] = appr.eval(0, loaders.total_tst_loader if not args.use_valid_only else loaders.total_val_loader, calculate_loss=False)
        test_loss, accuracies_task[t] = appr.eval(t, test_loader)
        forgetting_task_total[t] = accuracies_task_total[:t+1].max(0) - accuracies_task_total[t]
        forgetting_task[t] = accuracies_task[:t+1].max(0) - accuracies_task[t]

        accs_task_separate = np.zeros((max_task))
        for i in range(t+1):
            _, a = appr.eval(0, loaders.get_tst_loader_for_classes_only_in_t(i), calculate_loss=False)
            accs_task_separate[i] = a
            acc_triangle[t, i] = a

        print('>>> Test on all test samples :  | acc={:5.1f}%, forg={:5.1f}% <<<'.format( 100 * accuracies_task_total[t], 100 * forgetting_task_total[t],))
        print('>>> Test on task {:2d} : loss={:.3f} | acc={:5.1f}%, forg={:5.1f}% <<<'.format(t, test_loss, 100 * accuracies_task[t], 100 * forgetting_task[t],))
        print(f">>> Test on seperate task {' '.join('{:5.1f}% '.format(100 * a) for a in accs_task_separate)}")
        logger.log_scalar(task=t, iter=0, name='accuracies_task_total', group='test', value=100 * accuracies_task_total[t])
        logger.log_scalar(task=t, iter=0, name='forgetting_task_total', group='test', value=100 * forgetting_task_total[t])
        logger.log_scalar(task=t, iter=0, name='accuracies_task', group='test', value=100 * accuracies_task[t])
        logger.log_scalar(task=t, iter=0, name='forgetting_task', group='test', value=100 * forgetting_task[t])

        # Save
        print('Save at ' + os.path.join(args.results_path, full_exp_name))
        logger.log_result(accuracies_task_total, name="accuracies_task_total", step=t)
        logger.log_result(accuracies_task, name="forgetting_task", step=t)
        logger.log_result(forgetting_task_total, name="forgetting_task_total", step=t)
        logger.save_model(net.state_dict(), task=t)
        logger.log_result(accuracies_task_total.mean(), name="avg_accs_total", step=t)
        logger.log_result(accuracies_task.mean(), name="avg_accs", step=t)

        # Last layer analysis
        if args.last_layer_analysis:
            weights, biases = last_layer_analysis(net.head, t, loaders.num_classes, y_lim=True)
            logger.log_figure(name='weights', iter=t, figure=weights)
            logger.log_figure(name='bias', iter=t, figure=biases)

            # Output sorted weights and biases
            weights, biases = last_layer_analysis(net.head, t, loaders.num_classes, y_lim=True, sort_weights=True)
            logger.log_figure(name='weights', iter=t, figure=weights)
            logger.log_figure(name='bias', iter=t, figure=biases)
    logger.log_result(acc_triangle, "acc_triangle", step=max_task)
    # Print Summary
    utils.print_summary(accuracies_task, accuracies_task_total, forgetting_task_total)
    print('*' * 108)
    print("Acc Triangle")
    print(np.matrix(acc_triangle * 100))
    print('*' * 108)
    print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))
    print('Done!')

    return accuracies_task_total, forgetting_task_total, logger.exp_path
    ####################################################################################################################


if __name__ == '__main__':
    main()
