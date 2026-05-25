"""
Estimate rate coefficient by gradient descent of trajectory loss

Usage: python est_rate_traj.py -h
"""

from pathlib import Path
from pprint import pprint

import jax
import jax.numpy as jnp
import jaxtyping
import equinox as eqx
import optax
from optax import global_norm
import tqdm
import numpy as np

jax.config.update("jax_enable_x64", True)
FTYPE = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

import model
import data
import manager as mngr
from metrics import mse, log_mse
import plot

cfg = mngr.load_config()
pprint(cfg)

key = jax.random.PRNGKey(cfg["seed"])

if cfg["save_dir"]:
    ckpt_mngr = mngr.get_checkpoint_manager(
        cfg["save_dir"] / "checkpoints", cfg["ckpt_interval"], cfg["ckpt_keep"]
    )
    async_worker = mngr.get_async_worker(6)
    (cfg["save_dir"] / "logs").mkdir(parents=True, exist_ok=True)
    loss_logger = mngr.ScalarLogger(cfg["save_dir"] / "logs" / "loss.txt", async_worker)
    grad_norm_logger = mngr.ScalarLogger(
        cfg["save_dir"] / "logs" / "grad_norm.txt", async_worker
    )
    k_err_logger = mngr.ScalarLogger(
        cfg["save_dir"] / "logs" / "k_err.txt", async_worker
    )

# Data =========================================================================

sch, kinetics, ts, y0 = data.get_scheme(cfg["scheme"])
nspec = len(sch["SPC_NAMES"])
nrect = len(sch["EQN_NAMES"])

params = {
    **kinetics,
    "solver": cfg["solver"],
}

if cfg["scheme"] == "toy":
    b_ys_csv = data.load_toy_dataset(target_spc_names=sch["SPC_NAMES"])
    # re-compute trajectory to avoid numerical error between solvers
    b_ys = eqx.filter_vmap(model.solve, in_axes=(None, None, 0, None))(
        params, ts, b_ys_csv[:, 0, :], model.kinetic_ode
    )
elif cfg["scheme"] == "pollu":
    b_y0 = jnp.expand_dims(y0, 0)
    b_ys = eqx.filter_vmap(model.solve, in_axes=(None, None, 0, None))(
        params, ts, b_y0, model.kinetic_ode
    )
else:
    assert False, "Unknown scheme"

if cfg["obs_noise"]:
    b_ys = b_ys * (
        jnp.array(1 + cfg["obs_noise"], dtype=FTYPE)
        ** jax.random.normal(key, b_ys.shape)
    )
if cfg["obs_num"]:
    b_ys = b_ys[0 : cfg["obs_num"], :, :]
if cfg["obs_sample"]:
    sample_idx = jnp.linspace(0, ts.shape[0] - 1, num=cfg["obs_sample"], dtype=int)
    ts = ts[sample_idx]
    b_ys = b_ys[:, sample_idx, :]
    print("Using sample index: ", sample_idx)
if cfg["obs_scale"]:
    b_ys = b_ys * (1.0 + cfg["obs_scale"])

scale = {
    "yMax": jnp.max(b_ys, axis=(0, 1)),
    "yMin": jnp.min(b_ys, axis=(0, 1)),
    "tScale": jnp.array(ts[-1] - ts[0], dtype=FTYPE),
}
scale["yScale"] = jnp.where(
    scale["yMax"] - scale["yMin"] == 0.0, scale["yMax"], scale["yMax"] - scale["yMin"]
)
scale["ytScale"] = scale["yScale"] / scale["tScale"]

params["scale"] = scale


# Model ========================================================================

ground_truth = {
    "k_a_log": jnp.zeros(nrect, dtype=FTYPE),  # NOTE: log scale correction coefficients
}
var_params = ground_truth.copy()
fix_params = params


if cfg["k_noise"]:
    var_params["k_a_log"] = var_params["k_a_log"] + (
        jax.random.normal(key, var_params["k_a_log"].shape) * jnp.log(1 + cfg["k_noise"])
    )

fix_params["opt_mask"] = jax.tree.map(jnp.ones_like, var_params)

k_a_init = jnp.exp(var_params["k_a_log"])
k_a_true = jnp.exp(ground_truth["k_a_log"])


def _plot_k(k_a_pred, fname):
    fig = plot.plot_k(
        [k_a_pred, k_a_init, k_a_true],
        ["Estimated", "Initial", "Ground truth"],
    )
    fig.savefig(cfg["save_dir"] / fname)
    plot.plt.close(fig)


def _plot_y(traj_pred, fname):
    fig = plot.plot_series(y=traj_pred, t=ts, yy=b_ys[0], tt=ts)
    fig.savefig(cfg["save_dir"] / fname)
    plot.plt.close(fig)


def loss_fn(var_params, fix_params, ts, b_ys):
    params = {**var_params, **fix_params}
    b_ys_pred = eqx.filter_vmap(model.solve, in_axes=(None, None, 0, None))(
        params, ts, b_ys[:, 0], model.kinetic_correction_ode
    )
    loss = log_mse(b_ys_pred, b_ys)
    return loss


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
    grads = jax.tree.map(
        lambda g, m: jnp.where(m, g, 0.0), grads, fix_params["opt_mask"]
    )
    grad_norm = global_norm(eqx.filter(grads, eqx.is_inexact_array))
    updates, opt_state = optim.update(grads, opt_state, var_params, value=loss)
    new_var_params = eqx.apply_updates(var_params, updates)
    return new_var_params, loss, grad_norm, opt_state


# Training =====================================================================
optimizer = optax.chain(
    optax.clip(1.0),  # avoid huge gradient on chain init reaction
    optax.adam(cfg["learning_rate"]),
    optax.contrib.reduce_on_plateau(factor=0.5, patience=200),
)
opt_state = optimizer.init(eqx.filter(var_params, eqx.is_inexact_array))

start_epoch = 0
if cfg["resume"]:
    if cfg["resume_ckpt"]:
        restore_path = Path(cfg["resume_ckpt"]).absolute()
    else:
        assert cfg["save_dir"], "must provide save_dir or resume_ckpt to resume"
        restore_path = cfg["save_dir"] / "checkpoints"
    restore_mngr = mngr.get_checkpoint_manager(restore_path)
    start_epoch = restore_mngr.latest_step()
    print(f"Resume from checkpoint {restore_path}/{start_epoch}")
    abstract_state = {"var_params": var_params, "opt_state": opt_state}
    restored = mngr.standard_restore(restore_mngr, start_epoch, abstract_state)
    var_params = restored["var_params"]
    opt_state = restored["opt_state"]

length_strategy = [1.0]
epochs_strategy = [cfg["epochs"]]

if not cfg["train"]:
    epochs_strategy = []

for length, epochs in zip(length_strategy, epochs_strategy):
    print(f"strategy: length {length:.2f}, epoch {epochs:.2f}")

    bar = tqdm.tqdm(range(start_epoch + 1, epochs + 1), desc="Epochs", ncols=120)
    for i in bar:
        var_params, loss, grad_norm, opt_state = opt_step(
            optimizer, opt_state, var_params, fix_params, ts, b_ys
        )
        k_err = mse(var_params["k_a_log"], ground_truth["k_a_log"])
        bar.set_postfix(
            {
                "loss": f"{float(loss):.4e}",
                "gnorm": f"{float(grad_norm):.4e}",
                "k_err": f"{float(k_err):.4e}",
            }
        )

        if cfg["save_dir"]:  # checkpoint + scalar logging
            state = {"var_params": var_params, "opt_state": opt_state}
            mngr.standard_save(ckpt_mngr, i, state, loss)
            loss_logger.log(i, loss)
            grad_norm_logger.log(i, grad_norm)
            k_err_logger.log(i, k_err)

        if cfg["save_dir"] and i % cfg["test_interval"] == 0:  # testing
            params = {**var_params, **fix_params}
            traj_pred = model.solve(
                params, ts, b_ys[0, 0, :], model.kinetic_correction_ode
            )

            async_worker.submit(_plot_y, traj_pred, f"logs/est_ys_{i}.pdf")
            async_worker.submit(
                _plot_k, jnp.exp(var_params["k_a_log"]), f"logs/est_k_{i}.pdf"
            )

            loss_logger.flush()
            loss_logger.plot()
            grad_norm_logger.flush()
            grad_norm_logger.plot()
            k_err_logger.flush()
            k_err_logger.plot()


if cfg["infer"]:
    assert cfg["save_dir"], "must provide save_dir to load checkpoint"
    restore_step = ckpt_mngr.best_step()
    print(f"Inference using checkpoint {cfg['save_dir']}/checkpoints/{restore_step}")
    abstract_state = {"var_params": var_params, "opt_state": opt_state}
    state = mngr.standard_restore(ckpt_mngr, restore_step, abstract_state)

    k_a_pred = jnp.exp(state["var_params"]["k_a_log"])
    _plot_k(k_a_pred, "est_k.pdf")
    np.savez(
        cfg["save_dir"] / "est_k.npz",
        truth=k_a_true,
        pred=k_a_pred,
        init=k_a_init,
    )

    params = {**var_params, **fix_params}
    traj_pred = model.solve(params, ts, b_ys[0, 0, :], model.kinetic_correction_ode)
    _plot_y(traj_pred, "est_ys.pdf")

if cfg["save_dir"]:
    print("Finalising")
    loss_logger.flush()
    grad_norm_logger.flush()
    k_err_logger.flush()
    ckpt_mngr.wait_until_finished()
    ckpt_mngr.close()
    async_worker.shutdown()
