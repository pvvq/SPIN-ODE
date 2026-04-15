"""
Estimate rate coefficient by gradient descent of time derivative loss

Usage: python est_rate_dydt.py -h
"""

import sys
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import jaxtyping
import equinox as eqx
import optax
import tqdm
import numpy as np
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

import model
import data
import manager as mngr
from metrics import scale_mse, log_mse

sys.path.append(str(Path.cwd()))
import plots.plot as pp

FTYPE = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

cfg = mngr.load_config()
if cfg["save_dir"]:
    ckpt_mngr = mngr.get_checkpoint_manager(
        cfg["save_dir"] / "checkpoints",
        cfg["ckpt_interval"],
        cfg["ckpt_keep"],
    )

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
}

# prepare dydt data
if cfg["dydt"] == "neural_ode":
    neural_network = model.ScaleMLP(
        data_size=nspec,
        width_size=10,
        depth=4,
        key=jax.random.PRNGKey(cfg["seed"]),
    )

    # load checkpoint
    restore_path = Path(cfg["neural_ode_ckpt"]).absolute()
    restore_mngr = mngr.get_checkpoint_manager(restore_path)
    restore_step = restore_mngr.latest_step()
    print(f"Loading checkpoint {restore_path}/{restore_step}")
    abstract_state, static = eqx.partition(neural_network, eqx.is_array_like)
    restored_state = mngr.standard_restore(restore_mngr, restore_step, abstract_state)
    neural_network = eqx.combine(restored_state, static)

    params ["neural_network"] = neural_network
    b_dydt = eqx.filter_vmap(
        eqx.filter_vmap(model.neural_ode, in_axes=(None, 0, None)),
        in_axes=(None, 0, None)
    )(None, b_ys, params)
elif cfg["dydt"] == "finite_diff":
    b_dydt = jnp.gradient(b_ys, ts, axis=1)
else:
    assert False, f"{cfg["dydt"]} not defined"

# Model ========================================================================

var_names = ["k_static", "ro2_coef"]
ground_truth = {k: v for k, v in params.items() if k in var_names}
fix_params = {k: v for k, v in params.items() if k not in var_names}
var_params = jax.tree.map(jnp.zeros_like, ground_truth)  # NOTE: log scale
print(var_params)

def loss_fn(var_params, fix_params, dydt, ys):
    var_params = jax.tree.map(jnp.exp, var_params)
    params = {**var_params, **fix_params}

    def _pred(params, y):
        kinetic_dydt = model.kinetic_ode(None, y, params)
        return kinetic_dydt

    pred_dydt = eqx.filter_vmap(
        eqx.filter_vmap(_pred, in_axes=(None, 0)),
        in_axes=(None, 0),
    )(params, ys)
    loss = scale_mse(dydt, pred_dydt, params["scale"]["ytScale"])
    return loss


@eqx.filter_jit
def opt_step(
    optim: optax.GradientTransformation,
    opt_state: jaxtyping.PyTree,
    var_params,
    fix_params,
    dydt,
    ys,
):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(var_params, fix_params, dydt, ys)
    updates, opt_state = optim.update(grads, opt_state, var_params, value=loss)
    new_var_params = eqx.apply_updates(var_params, updates)
    return new_var_params, loss, opt_state


# Training =====================================================================
def train(var_params):
    optimizer = optax.chain(
        optax.clip(10),
        optax.adamw(cfg["learning_rate"]),
        # optax.contrib.reduce_on_plateau(factor=0.5, patience=100),
    )
    opt_state = optimizer.init(eqx.filter(var_params, eqx.is_inexact_array))

    length_strategy = [1.0]
    epochs_strategy = [cfg["epochs"]]
    for length, epochs in zip(length_strategy, epochs_strategy):
        print(f"strategy: length {length:.2f}, epoch {epochs:.2f}")

        bar = tqdm.tqdm(range(0, epochs), desc=f"Epochs", initial=0)
        for i in bar:
            var_params, loss, opt_state = opt_step(
                optimizer, opt_state, var_params, fix_params, b_dydt, b_ys
            )
            bar.set_postfix({"loss": f"{float(jnp.squeeze(loss)):.4e}"})

            if cfg["save_dir"]:  # checkpoint
                state = {"var_params": var_params, "opt_state": opt_state}
                mngr.standard_save(ckpt_mngr, i, state, loss)

    return var_params

if cfg["train"]:
    var_params = train(var_params)

import schemes.toy_autoxidation.rates as rates
def combine_static_ro2(tree):
    combined = jnp.zeros(rates.NREACT)
    combined = combined.at[rates._STATIC_DYN_INDICES].set(tree["k_static"][rates._STATIC_DYN_INDICES])
    combined = combined.at[rates._RO2_INDICES].set(tree["ro2_coef"])
    return combined


combined_true = combine_static_ro2(ground_truth)
combined_pred = combine_static_ro2(jax.tree.map(jnp.exp, var_params))

if cfg["save_dir"]:
    fig, ax = plt.subplots(1,1)
    ax.scatter(jnp.arange(combined_true.shape[0]), combined_true, label="true", marker="x")
    ax.scatter(jnp.arange(combined_pred.shape[0]), combined_pred, label="pred", marker="+")
    ax.legend()
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(cfg["save_dir"] / "est_k.pdf")

    np.savez(
        cfg["save_dir"] / "est_k.npz",
        truth=combined_true,
        pred=combined_pred,
    )

if cfg["save_dir"]:
    ckpt_mngr.wait_until_finished()
    ckpt_mngr.close()
