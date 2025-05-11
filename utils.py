import yaml
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def plot_series(traj, observe=None, t=None, t2=None, fig=None):
    """
    observe: [n_step, n_spc]
    traj: [n_step, n_spc]
    """
    n_species = traj.shape[1]
    cols = np.ceil(np.sqrt(n_species)).astype(int)
    rows = np.ceil(n_species / cols).astype(int)
    if fig is None: fig, ax = plt.subplots(
        rows, cols, figsize=(cols * 2, rows * 2),
        squeeze=False, layout='constrained',sharex=True
    )
    else:
        ax = np.array(fig.axes).reshape(rows, cols)
    if observe is not None:
        observe = observe.transpose(1,0)
    traj = traj.transpose(1,0)
    if t is None: t = range(traj.shape[1])
    if t2 is None and observe is not None: t2 = range(observe.shape[1])
    for i in range(n_species):
        ax[i%rows][i//rows].plot(t, traj[i])
        if observe is not None:
            ax[i%rows][i//rows].scatter(t2, observe[i],
                                        facecolors='none', edgecolors='black',
                                        s=10, linewidth=0.3)
        ax[i%rows][i//rows].set_title(i+1, fontsize='small', loc='left')
    # fig.savefig(f"{id}.png", dpi=300)
    return fig


def plot_err_k(true_k, k_list, label_list, num_react):
    fig, ax = plt.subplots()
    for i in range(len(k_list)):
        # symmetric mean absolute percentage error (SMAPE)
        err = np.abs(k_list[i] - true_k) / ((np.abs(k_list[i]) + np.abs(true_k)) / 2)
        ax.scatter(range(num_react), err, label=label_list[i])
    ax.set_yscale("linear")
    ax.legend()
    ax.set_xlabel("No. of reactions")
    ax.set_ylabel("Error of Rate Constant")
    # ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    return fig


def plot_k(k_list, label_list, num_react):
    marker_list = ['x', '+', '_']
    fig, ax = plt.subplots()
    for i in range(len(k_list)):
        ax.scatter(range(num_react), k_list[i], label=label_list[i], marker=marker_list[i])
    ax.set_yscale("log")
    ax.legend()
    ax.set_xlabel("No. of reactions")
    ax.set_ylabel("Rate Constant")
    return fig

class DummyLogger:
    def add_scalar(self, *args, **kwargs): pass
    def add_figure(self, *args, **kwargs): pass
    def add_text(self, *args, **kwargs): pass

class DummyCheckpointer:
    def save(self, *args, **kwargs): pass


class CMDArgsParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument("--config", "--cf", required=True, type=str, help="Path to YAML config")
        self.add_argument("--target", "-t", required=True, help="target in YAML config")
        self.add_argument('--log', action=argparse.BooleanOptionalAction, default=True, help="log to tensorboard")

def load_config():
    parser = CMDArgsParser()
    args = parser.parse_args()

    # Load YAML
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f).get(args.target)
        
    if config_dict is None:
        raise ValueError(f"Target '{target_key}' not found in config.")

    # Override YAML with non-None CLI args
    for key, value in vars(args).items():
        if value is not None:
            config_dict[key] = value

    return config_dict