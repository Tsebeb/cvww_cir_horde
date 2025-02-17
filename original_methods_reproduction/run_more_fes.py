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
            for model_name in ["resnet32", "slimresnet18"]:
                params = [sys.executable, "main_incremental.py", "--approach", approach,
                                "--datasets", "cifar100_autoaugment",
                                "--network", model_name,
                                "--num-tasks", "100",
                                "--loader-method", "cir_constant_probability",
                                '--pin-memory',
                                "--num-iterations-for-mean", "10",
                                "--seed", str(seed)]

                if approach == "horde":
                    for a in ["challenge", "confusion_matrix"]:
                        for num_fes in ["15", "20"]:
                            for larger_head in ["-1"]:
                                runs.append(params + ["--exp-name", f"more_fes_3rd_fe_{a}_{num_fes}_{larger_head}",
                                                      "--project-unk-mean", "feats",
                                                      "--fe-selection", a,
                                                      "--num-fe", num_fes,
                                                      "--head-latent-dim-size", larger_head,
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
