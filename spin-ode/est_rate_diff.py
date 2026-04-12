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

jax.config.update("jax_enable_x64", True)

import model
import data
import utils
from metrics import *

sys.path.append(str(Path.cwd()))
import plots.plot as pp

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
print(scale)

params = {
    **kinetics,
    "scale": scale,
}

# finite diff
b_dydt_finite = jnp.gradient(b_ys, ts, axis=1)


# neural ode
neural_network = model.ScaleMLP(
    data_size=nspec,
    width_size=10,
    depth=4,
    key=jax.random.PRNGKey(cfg["seed"]),
)

checkpointer = ocp.StandardCheckpointer()
ckpt_path = Path("checkpoints_cache/fit_all_toy").absolute()
# ckpt_path = ocp.test_utils.erase_and_create_empty(ckpt_path)
ckpt_idx = 4

# load checkpoint
state, static = eqx.partition(neural_network, eqx.is_array_like)
abstract_state = jax.tree.map(ocp.utils.to_shape_dtype_struct, state)
restored_state = checkpointer.restore(ckpt_path / f"{ckpt_idx}", abstract_state)
neural_network = eqx.combine(restored_state, static)

params ["neural_network"] = neural_network
b_dydt_nn = eqx.filter_vmap(
    eqx.filter_vmap(model.neural_ode, in_axes=(None, 0, None)),
    in_axes=(None, 0, None)
)(None, b_ys, params)

b_dydt = b_dydt_nn

# Model ========================================================================

var_names = ["k_static", "ro2_coef"]
ground_truth = {k: v for k, v in params.items() if k in var_names}
fix_params = {k: v for k, v in params.items() if k not in var_names}
var_params = jax.tree.map(jnp.zeros_like, ground_truth)
print(var_params)

def loss_fn(var_params, fix_params, dydt, ys):
    var_params = jax.tree.map(jnp.exp, var_params)
    params = {**var_params, **fix_params}

    def _pred(params, y):
        kinetic_dydt = model.kinetic_ode(None, y, params)
        return kinetic_dydt

    pred_dydt = eqx.filter_vmap(_pred, in_axes=(None, 0))(params, ys)
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
    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(var_params, fix_params, dydt, ys)
    updates, opt_state = optim.update(grads, opt_state, var_params)
    new_var_params = eqx.apply_updates(var_params, updates)
    return new_var_params, loss_val, opt_state


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
            var_params, loss_val, opt_state = opt_step(
                optimizer, opt_state, var_params, fix_params, b_dydt[0], b_ys[0]
            )
            bar.set_postfix(
                {
                    "loss": f"{float(jnp.squeeze(loss_val)):.4e}",
                }
            )

    with open("cache/est_rate_diff.pkl", "wb") as f:
        pickle.dump({
            "true": ground_truth,
            "pred": jax.tree.map(jnp.exp, var_params),
        }, f)

train(var_params)

with open("cache/est_rate_diff.pkl", "rb") as f:
    dump = pickle.load(f)
ground_truth, pred = dump["true"], dump["pred"]
print(ground_truth)
print(pred)

import schemes.toy_autoxidation.rates as rates
def combine_static_ro2(tree):
    combined = jnp.zeros(rates.NREACT)
    combined = combined.at[rates._STATIC_DYN_INDICES].set(tree["k_static"][rates._STATIC_DYN_INDICES])
    combined = combined.at[rates._RO2_INDICES].set(tree["ro2_coef"][rates._RO2_INDICES])
    return combined


import matplotlib.pyplot as plt
combined_true = combine_static_ro2(ground_truth)
combined_pred = combine_static_ro2(pred)
fig, ax = plt.subplots(1,1)
ax.scatter(jnp.arange(combined_true.shape[0]), combined_true, label="true", marker="x")
ax.scatter(jnp.arange(combined_pred.shape[0]), combined_pred, label="pred", marker="+")
ax.legend()
ax.set_yscale("log")
fig.tight_layout()
fig.savefig("plots/est_rate_diff.pdf")