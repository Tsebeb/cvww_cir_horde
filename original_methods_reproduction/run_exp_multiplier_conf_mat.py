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
    for seed in range(1):
        for approach in ["horde"]:
            params = [sys.executable, "main_incremental.py", "--approach", approach,
                            "--datasets", "cifar100_autoaugment",
                            "--network", "slimresnet18",
                            "--num-tasks", "100",
                            "--loader-method", "cir_constant_probability",
                            '--pin-memory',
                            "--num-iterations-for-mean", "10",
                            "--seed", str(seed)]

            if approach == "horde":
                for a in ["confusion_matrix"]:
                    for scale_factor in [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
                            runs.append(params + ["--exp-name", f"4th_experiment_scale_{scale_factor}",
                                                  "--project-unk-mean", "feats",
                                                  "--fe-selection", a,
                                                  "--fe-conf-strong-factor", str(scale_factor),
                                                  '--use-adaptive-alpha'])
            else:
                runs.append(params + ["--exp-name", f"first_exp_{seed}"])

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
