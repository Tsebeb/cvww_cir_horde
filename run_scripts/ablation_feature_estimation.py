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

    for s in ["303", "0", "1", "2", "3"]:
        for appr in ["initial_horde"]:
            if appr in ["initial_horde"]:
                for project_unk_mean in ["zeros", "noise", "feats"]:
                    total = ["1.0"] if project_unk_mean != "noise" else ["5.0", "25.0", "50.0", "75.0"]
                    for std in total:
                        runs.append(base_arguments + ["--approach", appr,
                                                      "--network", "slimresnet18",
                                                      "--exp-name", f"{exp_name}_estimation_{project_unk_mean}_with_{std}_seed_{s}",
                                                      "--fe-selection", "max_classes",
                                                      "--fe-lr", "0.001",
                                                      "--fe-epochs", "300",
                                                      "--project-unk-mean", project_unk_mean,
                                                      "--project_unk_std", std,
                                                      "--seed", s,
                                                      '--num-fe', "11"])
            else:
                raise RuntimeError("Method not Supported")


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
                                          "--loader-method", "cil",
                                          "--num-tasks", "11",
                                          "--num-start-classes", "50",
                                          "--validation", "0.1",
                                          "--num-workers", '16',
                                          "--batch-size", str(64),
                                          '--lr-patience', "5",
                                          '--pin-memory',
                                          "--lr", "0.001",
                                          "--nepochs", "300",
                                          "--momentum", str(0.9),
                                          "--weight-decay", str(0.0001),]
    run_methods(base_arguments_cil, parallel_processes, "ablation_estimate")


if __name__ == "__main__":
    experiment_cifar100()
