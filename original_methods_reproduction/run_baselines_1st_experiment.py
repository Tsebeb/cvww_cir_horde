import subprocess
import sys


def main():
    for a in ["finetuning", "ewc", "lwf"]:
        subprocess.run([sys.executable, "main_incremental.py", "--approach", a,
                        "--datasets", "cifar100_autoaugment",
                        "--network", "resnet32",
                        "--num-tasks", "100",
                        "--loader-method", "cir_constant_probability",
                        "--exp-name", "first_exp",
                        '--pin-memory',
                        "--save-models"])


if __name__ == "__main__":
    main()
