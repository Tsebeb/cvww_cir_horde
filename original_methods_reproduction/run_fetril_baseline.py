import subprocess
import sys
from multiprocessing.pool import ThreadPool
from typing import List

def process_run(args: List[str]):
    for a in args:
        if a == "":
            args.remove(a)
    return subprocess.run(args)


def main():
    parallel_processes = {"0": 2}
    runs_on_gpu = {key: [] for key in parallel_processes.keys()}
    runs = []

    for classifier_type in ["fc", "linear_svc"]:
        for use_l2_norm in ["", "--not-use-l2-norm"]:
            runs.append([sys.executable, "main_incremental.py",
                                "--approach", "fetril",
                                "--datasets", "cifar100_imagenet",
                                "--network", "resnet18",
                                "--num-tasks", "11",
                                "--loader-method", "cil",
                                "--num-start-classes", "50",
                                "--classifier-type", classifier_type,
                                '--pin-memory',
                                "--num-workers", '6',
                                "--nepochs", "20", "20", "10",
                                "--lr", "0.01", "0.001", "0.0001",
                                "--fe-data-augment", "cifar_imagenet_augmix",
                                "--test-augment", "cifar_imagenet_test",
                                "--lucir-pretrain-model", "resnet18_imagenet_lucir",
                                "--batch-size", str(64),
                                "--weight-decay", str(0.0001),
                                "--eval-on-train",
                                use_l2_norm,
                                "--exp-name", f"fetril_with_imagenet_size_{classifier_type}_{use_l2_norm}"])


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


if __name__ == "__main__":
    main()
