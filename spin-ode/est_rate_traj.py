# Usage: python traj_fit.py
#           --config <config_path>
#           --target <yaml_target>
#           --log/--no-log

import sys
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import jaxtyping
import equinox as eqx
import optax
import orbax.checkpoint as ocp
import tqdm
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

import model
import data
import utils
from metrics import *

sys.path.append(str(Path.cwd()))
import plots.plot as pp
import schemes.toy_autoxidation.rates as rates

FTYPE = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
cfg = utils.load_config()

# Data =========================================================================

sch, kinetics, ts, y0 = data.get_scheme("toy")
nspec = len(sch["SPC_NAMES"])

# params = {**kinetics, 'solver': cfg['solver']}
# ys = data.get_ys(params, ts, y0)

b_ys = data.load_toy_dataset(target_spc_names=sch["SPC_NAMES"])
# print(b_ys.shape)
ys = b_ys[0]
b_ys = jnp.expand_dims(ys, 0)


scale = {
    "yMax": jnp.max(b_ys, axis=(0, 1)),
    "yMin": jnp.min(b_ys, axis=(0, 1)),
    "tScale": jnp.array(ts[-1] - ts[0], dtype=FTYPE),
}
scale["yScale"] = jnp.where(
    scale["yMax"] - scale["yMin"] == 0.0, scale["yMax"], scale["yMax"] - scale["yMin"]
)
scale["ytScale"] = scale["yScale"] / scale["tScale"]


params = {
    **kinetics,
    "scale": scale,
    "solver": cfg["solver"],
}


# Model ========================================================================

var_names = ["k_static", "ro2_coef"]
ground_truth = {k: v for k, v in params.items() if k in var_names}
fix_params = {k: v for k, v in params.items() if k not in var_names}
# load predicted rate in step est_rate_diff
with open("cache/est_rate_diff.pkl", "rb") as f:
    dump = pickle.load(f)
var_params = jax.tree.map(jnp.log, dump["pred"])
print(var_params)

# only optimise NO and HO2 related reactions
fix_params["opt_mask"] = {
    'k_static': jnp.zeros(rates.NREACT, dtype=bool).at[4:12].set(True).at[20:28].set(True),
    'ro2_coef': jnp.zeros(rates.N_RO2, dtype=bool),
}

def combine_static_ro2(tree):
    combined = jnp.zeros(rates.NREACT)
    combined = combined.at[rates._STATIC_DYN_INDICES].set(tree["k_static"][rates._STATIC_DYN_INDICES])
    combined = combined.at[rates._RO2_INDICES].set(tree["ro2_coef"])
    return combined


def loss_fn(var_params, fix_params, ts, ys):
    var_params = jax.tree.map(jnp.exp, var_params)
    params = {**var_params, **fix_params}
    ys_pred = model.solve(params, ts, ys[0], model.kinetic_ode)
    return log_mse(ys_pred, ys)


@eqx.filter_jit
def opt_step(
    optim: optax.GradientTransformation,
    opt_state: jaxtyping.PyTree,
    var_params,
    fix_params,
    ts,
    ys,
):
    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(var_params, fix_params, ts, ys)
    grads = jax.tree.map(lambda g, m: jnp.where(m, g, 0.0), grads, fix_params["opt_mask"])
    updates, opt_state = optim.update(grads, opt_state, var_params, value=loss_val)
    new_var_params = eqx.apply_updates(var_params, updates)
    return new_var_params, loss_val, opt_state


# Training =====================================================================


def train(var_params):
    optimizer = optax.chain(
        optax.clip(1.0),  # avoid huge gradient on chain init reaction
        optax.adam(cfg["learning_rate"]),
        optax.contrib.reduce_on_plateau(factor=0.5, patience=200),
    )
    opt_state = optimizer.init(eqx.filter(var_params, eqx.is_inexact_array))

    length_strategy = [1.0]
    epochs_strategy = [cfg["epochs"]]
    for length, epochs in zip(length_strategy, epochs_strategy):
        print(f"strategy: length {length:.2f}, epoch {epochs:.2f}")

        bar = tqdm.tqdm(range(0, epochs), desc=f"Epochs", initial=0)
        for i in bar:
            var_params, loss_val, opt_state = opt_step(
                optimizer, opt_state, var_params, fix_params, ts, b_ys[0]
            )
            combined_true = combine_static_ro2(ground_truth)
            combined_pred = combine_static_ro2(jax.tree.map(jnp.exp, var_params))
            k_diff = log_mse(combined_pred, combined_true, eps=1e-16)
            bar.set_postfix(
                {
                    "loss": f"{float(jnp.squeeze(loss_val)):.4e}",
                    "k_diff": f"{float(jnp.squeeze(k_diff)):.4e}",
                }
            )

            if i % 100 == 0:
                # Testing
                traj_pred = model.solve(
                    {**jax.tree.map(jnp.exp, var_params), **fix_params}, ts, ys[0], model.kinetic_ode
                )
                fig = pp.plot_series(y=traj_pred, t=ts, yy=ys, tt=ts)
                fig.savefig("plots/est_traj.pdf")

                fig, ax = plt.subplots(1,1)
                ax.scatter(jnp.arange(combined_true.shape[0]), combined_true, label="true", marker="x")
                ax.scatter(jnp.arange(combined_pred.shape[0]), combined_pred, label="pred", marker="+")
                ax.legend()
                ax.set_yscale("log")
                fig.tight_layout()
                fig.savefig("plots/est_rate_traj.pdf")

            if i % 500 == 0:
                # checkpoint
                with open("cache/est_rate_traj.pkl", "wb") as f:
                    pickle.dump({
                        "true": ground_truth,
                        "pred": jax.tree.map(jnp.exp, var_params),
                    }, f)

train(var_params)

with open("cache/est_rate_traj.pkl", "rb") as f:
    dump = pickle.load(f)
ground_truth, pred = dump["true"], dump["pred"]
print(ground_truth)
print(pred)