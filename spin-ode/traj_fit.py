"""
Neural ODE fit trajectory

Usage: python traj_fit.py -h
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxtyping
import equinox as eqx
import optax
import tqdm

jax.config.update("jax_enable_x64", True)

import model
import data
import manager as mngr
from metrics import scale_mse

sys.path.append(str(Path.cwd()))
import plot

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

# Model ========================================================================

neural_network = model.ScaleMLP(
    data_size=nspec,
    width_size=10,
    depth=4,
    key=jax.random.PRNGKey(cfg["seed"]),
)

var_params = {
    "neural_network": neural_network,
}
fix_params = {"solver": cfg["solver"], "scale": scale}


def loss_fn_b(var_params, fix_params, ts, b_ys):
    params = {**var_params, **fix_params}
    ys_pred = eqx.filter_vmap(model.solve, in_axes=(None, None, 0, None))(
        params, ts, b_ys[:, 0], model.neural_ode
    )
    return scale_mse(ys_pred, b_ys, params["scale"]["yScale"])


def loss_fn(var_params, fix_params, ts, ys):
    params = {**var_params, **fix_params}
    ys_pred = model.solve(params, ts, ys[0], model.neural_ode)
    return scale_mse(ys_pred, ys, params["scale"]["yScale"])


@eqx.filter_jit
def opt_step(
    optim: optax.GradientTransformation,
    opt_state: jaxtyping.PyTree,
    var_params,
    fix_params,
    ts,
    ys,
):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(var_params, fix_params, ts, ys)
    updates, opt_state = optim.update(grads, opt_state, var_params)
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

        bar = tqdm.tqdm(range(0, epochs), desc="Epochs", initial=0)
        for i in bar:
            var_params, loss, opt_state = opt_step(
                optimizer, opt_state, var_params, fix_params, ts, ys
            )
            bar.set_postfix(
                {
                    "loss": f"{float(jnp.squeeze(loss)):.4e}",
                }
            )

            if cfg["save_dir"]:  # checkpoint
                state, _ = eqx.partition(neural_network, eqx.is_array_like)
                mngr.standard_save(ckpt_mngr, i, state, loss)

            if cfg["save_dir"] and i % cfg["test_interval"] == 0:  # Testing
                traj_pred = model.solve(
                    {**var_params, **fix_params}, ts, ys[0], model.neural_ode
                )
                err_traj = scale_mse(traj_pred, ys, scale["yScale"])
                fig = plot.plot_series(y=traj_pred, t=ts, yy=ys, tt=ts)
                fig.savefig(cfg["save_dir"] / "traj_fit.pdf")

    return var_params


if cfg["train"]:
    var_params = train(var_params)

if cfg["save_dir"]:
    ckpt_mngr.wait_until_finished()
    ckpt_mngr.close()
