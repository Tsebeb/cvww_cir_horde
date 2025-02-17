import os
import subprocess
import sys
from multiprocessing.pool import ThreadPool
from typing import List


def process_run(args: List[str]):
    for a in args:
        if a == "":
            args.remove(a)
    return subprocess.run(args, cwd=os.path.join("..", "src"))


def run_methods(base_arguments, parallel_processes, exp_name):
    runs_on_gpu = {key: [] for key in parallel_processes.keys()}
    runs = []

    for random_seed in ["0", "1", "2", "3", "303"]:
        custom_arguments = base_arguments + ["--seed", random_seed]
        for appr in ["finetuning", "freezing", "ewc", "lwf", "mas", "il2a", "pass", "ssre", "wa", "fetril", "praka", "initial_horde"]:
            # setup base methods
            if appr in ["finetuning", "freezing"]:
                runs.append(custom_arguments + ["--approach", appr,
                                              "--network", "resnet18_cbam",
                                              "--exp-name", f"{exp_name}_seed_{random_seed}",
                                              "--task-ce-only"])
            elif appr in ["il2a", "pass", "praka", "joint"]:
                runs.append(custom_arguments + ["--approach", appr,
                                              "--network", "resnet18_cbam",
                                              "--exp-name", f"{exp_name}_seed_{random_seed}"])
            elif appr in ["ewc"]:
                runs.append(custom_arguments + ["--approach", appr, "--network",
                                              "resnet18_cbam",
                                              "--exp-name", f"{exp_name}_seed_{random_seed}",
                                              "--lamb", "40000", "--task-ce-only",
                                              "--alpha", "0.1"])
            elif appr in ["mas"]:
                runs.append(custom_arguments + ["--approach", appr, "--network",
                                              "resnet18_cbam",
                                              "--exp-name", f"{exp_name}_seed_{random_seed}",
                                              "--lamb", "10.0", "--task-ce-only",
                                              "--alpha", "0.1"])
            elif appr in ["lwf"]:
                runs.append(custom_arguments + ["--approach", appr,
                                              "--network", "resnet18_cbam",
                                              "--exp-name", f"{exp_name}_seed_{random_seed}",
                                              "--lamb", str(30.0)])
            elif appr in ["fetril"]:
                runs.append( custom_arguments + ["--approach", appr,
                                               "--network", "resnet18_cbam",
                                               "--lucir-pretrain-model", "none",
                                               "--fe-data-augment", "cifar_augmix",
                                               "--fe-lr", "0.001",
                                               "--fe-epochs", "300",
                                               "--lr-patience", "10",
                                               "--test-augment", "train",
                                               "--exp-name", f"{exp_name}_seed_{random_seed}"])
            elif appr in ["wa"]:
                runs.append(custom_arguments + ["--approach", appr,
                                              "--network", "resnet18_cbam",
                                              "--num-exemplars", "2000",
                                              "--initial-nepochs", "300",
                                              "--initial-lr", "0.001",
                                              "--lr-patience", "10",
                                              "--initial-wd", str(0.0001),
                                              "--exp-name", f"{exp_name}_seed_{random_seed}"])
            elif appr in ["ssre"]:
                runs.append(custom_arguments + ["--approach", appr,
                                              "--network", "resnet18_ssre_bn",
                                              "--exp-name", f"{exp_name}_seed_{random_seed}"])
            elif appr in ["horde"]:
                for feat_selection in ["feats"]:
                    for num_fe in ["10"]:
                        runs.append(custom_arguments + ["--approach", appr,
                                                      "--network", "slimresnet18",
                                                      "--exp-name", f"{exp_name}_sel_{feat_selection}_fe_{num_fe}_seed_{random_seed}",
                                                      "--fe-selection", "confusion_matrix",
                                                      "--fe-lr", "0.001",
                                                      "--fe-epochs", "300",
                                                      '--num-fe', num_fe,
                                                      "--use-self-supervision"])
            elif appr in ["initial_horde"]:
                for fe_selection in ["max_classes", "confusion_matrix"]:
                    runs.append(custom_arguments + ["--approach", appr,
                                                  "--network", "slimresnet18",
                                                  "--exp-name", f"{exp_name}_sel_{fe_selection}_seed_{random_seed}",
                                                  "--fe-selection", fe_selection,
                                                  "--fe-lr", "0.001",
                                                  "--fe-epochs", "300",
                                                  '--num-fe', str(10),
                                                  "--use-self-supervision", "--acc-prototype",
                                                  "--initial-network-name", "resnet18_cbam"])
            elif appr in ["plastil"]:
                for num_tops in ["5"]:
                    runs.append(base_arguments + ["--approach", appr,
                                                  "--network", "resnet18_cbam",
                                                  "--exp-name", f"{exp_name}_tops_{num_tops}",
                                                  "--num-model-tops", num_tops])

            else:
                raise RuntimeError("")


    # Split remaining runs on available GPUS
    num_gpus_available = len(runs_on_gpu.keys())
    gpu_ids = list(runs_on_gpu.keys())
    for i in range(len(runs)):
        gpu_id = gpu_ids[i % num_gpus_available]
        runs_on_gpu[gpu_id].append(runs[i] + ["--gpu", str(gpu_id)])

    pools: List[ThreadPool] = []
    for p in parallel_processes:
        pools.append(ThreadPool(processes=parallel_processes[p]))

    for i, k in enumerate(runs_on_gpu):
        pools[i].map_async(process_run, runs_on_gpu[k])

    # Wait until all pools have finished
    for pool in pools:
        pool.close()
        pool.join()


def experiment_cifar100():
    parallel_processes = {"0": 1}
    base_arguments_cil = [sys.executable, "main_incremental.py",
                                          "--datasets", "cifar100_il2a",
                                          "--loader-method", "cir_beta_frequency",
                                          "--num-tasks", "100",
                                          "--num-start-classes", "50",
                                          "--num-start-samples", "225",
                                          "--validation", "0.1",
                                          "--num-workers", '0',
                                          "--batch-size", str(64),
                                          '--lr-patience', "5",
                                          '--pin-memory',
                                          "--lr", "0.001",
                                          "--nepochs", "300",
                                          "--momentum", str(0.9),
                                          "--weight-decay", str(0.0001),]
    run_methods(base_arguments_cil, parallel_processes, "efcir_beta_more_seeds")


if __name__ == "__main__":
    experiment_cifar100()
