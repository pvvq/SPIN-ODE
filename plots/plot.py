import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


def plot_series(
        y: npt.ArrayLike,
        t = None,
        yy: npt.ArrayLike = None,
        tt = None,
        fig = None,
        grid: tuple = None,
    ):
    """
    Args:
        y: [n_step, n_spc]
        t: [n_step]
        yy: [n_step, n_spc]
        tt: [n_step]
        fig: plot on top of existing fig
        grid: (rows, cols) of axes
    """
    n_species = y.shape[-1]
    if grid:
        rows, cols = grid
    else:
        cols = np.ceil(np.sqrt(n_species)).astype(int)
        rows = np.ceil(n_species / cols).astype(int)
    if fig is None: 
        fig, axes = plt.subplots(
            rows, cols, figsize=(cols * 2, rows * 2),
            squeeze=False, layout='constrained',
            sharex=True,
        )
    else:
        axes = np.array(fig.axes).reshape(rows, cols)
    
    y = y.transpose(1,0)
    if yy is not None:
        yy = yy.transpose(1,0)
    if t is None: t = range(y.shape[1])
    if tt is None and yy is not None: tt = range(yy.shape[1])

    for i in range(n_species):
        ax = axes[i//cols][i%cols]
        ax.plot(t, y[i])
        if yy is not None:
            ax.scatter(
                tt, yy[i],
                facecolors='none', edgecolors='black',
                s=10, linewidth=0.3
            )
        ax.set_title(i+1, fontsize='small', loc='left')
        
    fig.supxlabel("t")
    fig.supylabel("y", rotation=0)

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