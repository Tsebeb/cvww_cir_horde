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

    for ds in ["cifar100"]:
        for clip_weights in ["--no-positive-weight-clipping"]:
            runs.append([sys.executable, "main_incremental.py",
                                "--approach", "wa",
                                "--datasets", ds,
                                "--network", "resnet18_wa_cbam",
                                "--num-tasks", "10",
                                "--loader-method", "cil",
                                "--num-start-classes", "10",
                                '--pin-memory',
                                clip_weights,
                                "--num-workers", '0',
                                "--initial-nepochs", "42", "18", "20", "20",
                                "--initial-lr", "0.1", "0.01", "0.001", "0.0001",
                                "--nepochs", "42", "18", "20", "20",
                                "--lr", "0.1", "0.01", "0.001", "0.0001",
                                "--batch-size", str(64),
                                "--weight-decay", str(0.0002),
                                "--initial-wd", str(0.0002),
                                "--momentum", str(0.9),
                                "--num-exemplars", "2000",
                                "--exp-name", f"baseline_unofficial_4",
                                "--seed", "2222"])

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
