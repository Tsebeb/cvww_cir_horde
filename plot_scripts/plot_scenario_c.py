import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tabulate
from matplotlib import rcParams

lookup_table = {
            'finetuning': {'name': 'FT', 'color': '#8CD17D', 'linestyle': 'dashed', 'table_name': "\\gls{ft}"},
            'horde': {'name': 'Horde', 'color': '#9edae5', 'linestyle': 'solid', 'table_name': "\\gls{horde}"},
            'ewc': {'name': 'EWC', 'color': '#666666', 'linestyle': 'dashed', 'table_name': "\\gls{ewc}"},
            'lwf': {'name': 'LwF', 'color': '#ff7f0e', 'linestyle': 'dashed', 'table_name': "\\gls{lwf}"},
            'mas': {'name': 'MAS', 'color': '#ff9896', 'linestyle': 'dashed', 'table_name': "\\gls{mas}"},
            'il2a': {'name': 'IL2A', 'color': '#e39802', 'linestyle': 'dotted', 'table_name': "\\gls{il2a}"},
            'pass': {'name': 'PASS', 'color': '#d62728', 'linestyle': 'dotted', 'table_name': "\\gls{pass}"},
            'ssre': {'name': 'SSRE', 'color': '#cc78bc', 'linestyle': 'dotted', 'table_name': "\\gls{ssre}"},
            'fetril': {'name': 'FeTrIL', 'color': '#ffda66', 'linestyle': 'dotted', 'table_name': "\\gls{fetril}"},
            'plastil': {'name': 'PlaStIL', 'color': '#56b4e9', 'linestyle': 'dotted', 'table_name': "\\gls{plastil}"},
            'wa': {'name': 'WA', 'color': '#fbafe4', 'linestyle': 'dashdot', 'table_name': "\\gls{wa}"},
            'praka': {'name': 'PRAKA', 'color': '#8c564b', 'linestyle': 'dotted', 'table_name': "\\gls{praka}"},
            'freezing': {'name': 'FZ', 'color': '#949494', 'linestyle': 'dashed', 'table_name': "\\gls{fz}"},
            'joint': {'name': 'Joint', 'color': '#0173b2', 'linestyle': 'dashdot', 'table_name': "\\gls{join}"},
            "ihorde_max_classes": {'name': '$Horde_{m}$', 'color': '#029e73', 'linestyle': 'solid', 'table_name': "$\\gls{horde}_m$"},
            'ihorde_confusion_matrix': {'name': '$Horde_{c}$', 'color': '#9edae5', 'linestyle': 'solid', 'table_name': "$\\gls{horde}_m$"},
        }

print_order_legend = {"FT": 0,
                      "FZ": 1,
                      "EWC": 2,
                      "MAS": 3,
                      "LwF": 4,
                      "PASS": 5,
                      "PRAKA": 6,
                      "IL2A": 7,
                      "SSRE": 8,
                      "FeTrIL": 9,
                      "WA": 10,
                      " ": 11,
                      "$Horde_{m}$": 12,
                      "$Horde_{c}$": 13, }

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
    ax = fig.add_axes([0.075, 0.1, 0.9, 0.8])
    ax.set_title(f"{dataset_name} Classes seen" if title is None else title, y=1.0, pad=-18, fontsize=16)

    grouped_e = {}
    for e in experiments:
        sout = get_latest_stdout(e)
        args = get_latest_args(e)
        if "initial_horde" in os.path.basename(e):
            approach_name = "ihorde_" + args["fe_selection"]
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
    ax.plot([0], [0], "-", color="none", label=" ")
    handles, labels = plt.gca().get_legend_handles_labels()
    new_handles, new_labels = [None] * 14, [None] * 14
    for i, l in enumerate(labels):
        lookup_idx = print_order_legend[l]
        new_handles[lookup_idx] = handles[i]
        new_labels[lookup_idx] = l

    ax.legend(new_handles, new_labels, bbox_to_anchor=(0.5, 1.135), loc='upper center', ncol=7, frameon=False)
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.set_xlabel("Task", fontsize=14)
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
    rcParams["font.size"] = 12
    rcParams['lines.linewidth'] = 2

    plot_more_seeds("cifar100_il2a", "beta_more_seeds", "EFCIR Beta",
                    save_name="efcir_beta.png",)
    sys.exit(0)
