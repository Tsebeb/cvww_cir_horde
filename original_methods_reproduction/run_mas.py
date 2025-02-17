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
    parallel_processes = {"0": 5}
    runs_on_gpu = {key: [] for key in parallel_processes.keys()}
    runs = []
    for seed in range(5):
        for approach in ["mas"]:
            params = [sys.executable, "main_incremental.py", "--approach", approach,
                            "--datasets", "cifar100_autoaugment",
                            "--network", "resnet18_cifar",
                            "--num-tasks", "100",
                            "--loader-method", "cir_constant_probability",
                            '--pin-memory',
                            "--seed", str(seed)]
            runs.append(params + ["--exp-name", f"mas_exp_seed_{seed}"])

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
