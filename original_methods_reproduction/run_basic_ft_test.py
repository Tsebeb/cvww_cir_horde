import subprocess
import sys
from multiprocessing.pool import ThreadPool
from typing import List

def process_run(args: List[str]):
    for a in args:
        if a == "":
            args.remove(a)
    return subprocess.run(args)

# Seed 303 is the best with 78,8% acc!
def main():
    parallel_processes = {"0": 3}
    runs_on_gpu = {key: [] for key in parallel_processes.keys()}
    runs = []

    for seed in range(1):
        runs.append([sys.executable, "main_incremental.py",
                            "--approach", "temp_ft",
                            "--datasets", "cifar100_il2a",
                            "--network", "resnet18_cifar",
                            "--num-tasks", "11",
                            "--loader-method", "cil",
                            "--num-start-classes", "50",
                            '--pin-memory',
                            "--num-workers", '0',
                            "--nepochs", "45", "45", "11",
                            "--lr", "0.001", "0.0001", "0.00001",
                            "--batch-size", str(64),
                            "--weight-decay", str(1e-4),
                            "--exp-name", f"temp_ft_seed_{seed}",
                            "--stop-at-task", "1",
                            "--seed", str(seed)])

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
