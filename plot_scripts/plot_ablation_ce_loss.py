import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tabulate
from matplotlib import rcParams

lookup_table = {
            'finetuning': {'name': 'FT - Full CE', 'color': '#8CD17D', 'linestyle': 'solid', 'table_name': "\\gls{ft} - full ce"},
            'ewc': {'name': 'EWC - Full CE', 'color': '#666666', 'linestyle': 'solid', 'table_name': "\\gls{ewc} - full ce"},
            'mas': {'name': 'MAS - Full CE', 'color': '#ff9896', 'linestyle': 'solid', 'table_name': "\\gls{mas} - full ce"},

            'finetuning_task': {'name': 'FT - Task CE', 'color': '#8CD17D', 'linestyle': 'dashed', 'table_name': "\\gls{ft} - task ce"},
            'ewc_task': {'name': 'EWC - Task CE', 'color': '#666666', 'linestyle': 'dashed', 'table_name': "\\gls{ewc} - task ce"},
            'mas_task': {'name': 'MAS - Task CE', 'color': '#ff9896', 'linestyle': 'dashed', 'table_name': "\\gls{mas} - task ce"},

        }

def get_latest_stdout(results_path):
    files = os.listdir(results_path)
    files = [f for f in files if "stdout" in f]
    files.sort()
    latest_files = files[-1]
    with open(os.path.join(results_path, latest_files), "r") as f:
        return f.readlines()


def get_latest_args(results_path):
    files = os.listdir(results_path)
    files = [f for f in files if "args" in f]

    files.sort()
    latest_files = files[-1]

    with open(os.path.join(results_path, latest_files), "r") as f:
        return json.load(f)


def get_latest_error(results_path):
    files = os.listdir(results_path)
    files = [f for f in files if "stderr" in f]
    files.sort()
    latest_files = files[-1]
    with open(os.path.join(results_path, latest_files), "r") as f:
        return f.readlines()


def get_latest_result(results_path, result_array_name):
    results_path = os.path.join(results_path, "results")
    files = os.listdir(results_path)
    files = [f for f in files if result_array_name in f]
    files.sort()
    latest_files = files[-1]

    with open(os.path.join(results_path, latest_files), "r") as f:
        return np.loadtxt(f, delimiter='\t')


def check_experiment_for_error(e):
    error_lines = get_latest_error(e)
    for l in error_lines:
        if "error" in l.lower():
            return True
    return False

def check_is_done(results_text):
    for line in results_text[-20:]:
        if "Done!" in line:
            return True
    return False


def plot_more_seeds(dataset_name, exp_search_string, title=None, save_name=None, plot_std=True, methods_highlight=None):
    list_path = "./"
    experiments = [os.path.join(list_path, f) for f in os.listdir(list_path) if dataset_name in f and exp_search_string in f]

    highlight_alpha = 0.7
    shaded_highlight = 0.15
    other_alpha = 0.1
    shaded_other = 0.1

    if methods_highlight is None:
        other_alpha = highlight_alpha
        shaded_other = shaded_highlight
        methods_highlight = []

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes([0.1, 0.135, 0.875, 0.7])
    # ax.set_title(f"{dataset_name} Classes seen" if title is None else title, y=1.0, pad=-18, fontsize=16)

    grouped_e = {}
    for e in experiments:
        sout = get_latest_stdout(e)
        args = get_latest_args(e)
        if "task_ce" in os.path.basename(e):
            approach_name = os.path.basename(e).replace(dataset_name, "").split("_")[1] + "_task"
        else:
            approach_name = os.path.basename(e).replace(dataset_name, "").split("_")[1]
        if check_experiment_for_error(e):
            print(f"The experiment {e} has encountered an error skipping for now")
            continue

        if not check_is_done(sout):
            continue

        if approach_name not in grouped_e:
            grouped_e[approach_name] = [e]
        else:
            grouped_e[approach_name].append(e)

    table = []
    for approach_name in grouped_e:
        accs = []
        for e in grouped_e[approach_name]:
            seed_acc = get_latest_result(e, "forgetting_task-")  # wrong name but i am not changing it now :)
            accs.append(seed_acc)

        accs = np.array(accs)
        label = lookup_table[approach_name]["name"]
        color = lookup_table[approach_name]["color"]
        linestyle = lookup_table[approach_name]["linestyle"]
        is_highlight = approach_name in methods_highlight
        x_label = np.arange(accs.shape[1])

        # Filter bad runs
        seed_mean = np.mean(accs, axis=1)
        median_seed = np.median(seed_mean)
        mask = np.abs(seed_mean - median_seed) < 0.15
        accs = accs[mask]

        mean_accs = np.mean(accs, axis=0)
        std_accs = np.std(accs, axis=0)
        ax.plot(x_label, mean_accs, label=label,  color=color, alpha=highlight_alpha if is_highlight else other_alpha,
                linestyle=linestyle, zorder=2)
        if plot_std:
            ax.fill_between(x_label, mean_accs - std_accs, mean_accs + std_accs, color=color,
                            alpha=shaded_highlight if is_highlight else shaded_other, zorder=0, linewidth=0.0)

        table.append([approach_name, np.mean(mean_accs), np.mean(std_accs), mean_accs[-1]])

    # Custom resorting of labels
    # add placeholder_element for legend
    # ax.plot([0], [0], "-", color="none", label=" ")
    handles, labels = plt.gca().get_legend_handles_labels()
    # new_handles, new_labels = [None] * 14, [None] * 14
    # for i, l in enumerate(labels):
    #     lookup_idx = print_order_legend[l]
    #     new_handles[lookup_idx] = handles[i]
    #     new_labels[lookup_idx] = l

    ax.legend(handles, labels, bbox_to_anchor=(0.5, 1.225), loc='upper center', ncol=3, frameon=False)
    ax.set_ylabel("Accuracy", fontsize=24)
    ax.set_xlabel("Task", fontsize=24)
    ax.set_xlim((-1, 101))
    ax.set_ylim((0.0, 0.75))
    ax.set_xticks(list(range(0, 101, 10)), minor=False)
    ax.set_xticks(list(range(5, 101, 10)), minor=True)
    if save_name is not None:
        plt.savefig(save_name)
    plt.show()
    plt.clf()

    table = sorted(table, key=lambda x: x[1], reverse=True)
    print(tabulate.tabulate(table, headers=["Method", "Mean Avg. Accs", "Mean Std. Accs", "Final Accs"]))


if __name__ == "__main__":
    os.chdir(os.path.join("..", "results"))
    rcParams["figure.dpi"] = 200.0
    rcParams['font.family'] = "Times New Roman"
    rcParams["font.size"] = 18
    rcParams['lines.linewidth'] = 2

    plot_more_seeds("cifar100_il2a", "efcir_ablation_head", "EFCIR uniform",
                    save_name="ablation_task.png",)
    sys.exit(0)
