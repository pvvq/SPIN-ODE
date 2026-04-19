"""
Estimate rate coefficient by gradient descent of time derivative loss

Usage: python est_rate_dydt.py -h
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import jaxtyping
import equinox as eqx
import optax
from optax import global_norm
import tqdm
import numpy as np

jax.config.update("jax_enable_x64", True)

import model
import data
import manager as mngr
from metrics import scale_mse
import plot

FTYPE = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

cfg = mngr.load_config()
if cfg["save_dir"]:
    ckpt_mngr = mngr.get_checkpoint_manager(
        cfg["save_dir"] / "checkpoints",
        cfg["ckpt_interval"],
        cfg["ckpt_keep"],
    )
    async_worker = mngr.get_async_worker(6)
    (cfg["save_dir"] / "logs").mkdir(parents=True, exist_ok=True)
    loss_logger = mngr.ScalarLogger(cfg["save_dir"] / "logs" / "loss.txt", async_worker)
    grad_norm_logger = mngr.ScalarLogger(
        cfg["save_dir"] / "logs" / "grad_norm.txt", async_worker
    )

# Data =========================================================================

sch, kinetics, ts, y0 = data.get_scheme("toy")
nspec = len(sch["SPC_NAMES"])

params = {
    **kinetics,
    "solver": cfg["solver"],
}

b_ys_csv = data.load_toy_dataset(target_spc_names=sch["SPC_NAMES"])
# re-compute trajectory to avoid numerical error between solvers
b_ys = eqx.filter_vmap(model.solve, in_axes=(None, None, 0, None))(
    params, ts, b_ys_csv[:, 0, :], model.kinetic_ode
)

if cfg["obs_num"]:
    b_ys = b_ys[0 : cfg["obs_num"], :, :]
if cfg["obs_sample"]:
    sample_idx = jnp.linspace(
        0, ts.shape[0], num=cfg["obs_sample"], endpoint=False, dtype=int
    )
    ts = ts[sample_idx]
    b_ys = b_ys[:, sample_idx, :]
    print("Using sample index: ", sample_idx)
if cfg["obs_noise"]:
    subkey, key = jax.random.split(key, 2)
    b_ys = data.add_normal_noise(b_ys, float(cfg["obs_noise"]), subkey)



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

# prepare dydt data
if cfg["dydt"] == "neural_ode":
    neural_network = model.ScaleMLP(
        data_size=nspec,
        width_size=10,
        depth=4,
        key=jax.random.PRNGKey(cfg["seed"]),
    )

    # load checkpoint
    # TODO: now retore old state, which is nn only, retrain and store use nn+opt_step
    restore_path = Path(cfg["neural_ode_ckpt"]).absolute()
    restore_mngr = mngr.get_checkpoint_manager(restore_path)
    restore_step = restore_mngr.best_step()
    print(f"Loading checkpoint {restore_path}/{restore_step}")
    abstract_state, static = eqx.partition(neural_network, eqx.is_array_like)
    restored_state = mngr.standard_restore(restore_mngr, restore_step, abstract_state)
    neural_network = eqx.combine(restored_state, static)

    params["nn"] = neural_network
    b_dydt = eqx.filter_vmap(
        eqx.filter_vmap(model.neural_ode, in_axes=(None, 0, None)),
        in_axes=(None, 0, None),
    )(None, b_ys, params)
elif cfg["dydt"] == "finite_diff":
    b_dydt = jnp.gradient(b_ys, ts, axis=1)
else:
    assert False, f"{cfg['dydt']} not defined"

# Model ========================================================================

var_names = ["k_static", "ro2_coef"]
ground_truth = {k: v for k, v in params.items() if k in var_names}
fix_params = {k: v for k, v in params.items() if k not in var_names}
var_params = jax.tree.map(jnp.zeros_like, ground_truth)  # NOTE: log scale
print(var_params)

combined_true = data.combine_static_ro2(ground_truth)
combined_init = data.combine_static_ro2(jax.tree.map(jnp.exp, var_params))


def _plot_k(var_params, fname):
    combined_pred = data.combine_static_ro2(jax.tree.map(jnp.exp, var_params))
    fig = plot.plot_k(
        [combined_pred, combined_init, combined_true],
        ["Estimated", "Initial", "Ground truth"],
    )
    fig.savefig(cfg["save_dir"] / fname)
    plot.plt.close(fig)


def loss_fn(var_params, fix_params, dydt, ys):
    var_params = jax.tree.map(jnp.exp, var_params)
    params = {**var_params, **fix_params}

    def _pred_dydt(params, y):
        kinetic_dydt = model.kinetic_ode(None, y, params)
        return kinetic_dydt

    dydt_pred = eqx.filter_vmap(
        eqx.filter_vmap(_pred_dydt, in_axes=(None, 0)),
        in_axes=(None, 0),
    )(params, ys)
    loss = scale_mse(dydt, dydt_pred, params["scale"]["ytScale"])
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
    grad_norm = global_norm(eqx.filter(grads, eqx.is_inexact_array))
    updates, opt_state = optim.update(grads, opt_state, var_params, value=loss)
    new_var_params = eqx.apply_updates(var_params, updates)
    return new_var_params, loss, grad_norm, opt_state


# Training =====================================================================
optimizer = optax.chain(
    optax.clip(10),
    optax.adamw(cfg["learning_rate"]),
    # optax.contrib.reduce_on_plateau(factor=0.5, patience=100),
)
opt_state = optimizer.init(eqx.filter(var_params, eqx.is_inexact_array))


length_strategy = [1.0]
epochs_strategy = [cfg["epochs"]]

if not cfg["train"]:
    epochs_strategy = []

for length, epochs in zip(length_strategy, epochs_strategy):
    print(f"strategy: length {length:.2f}, epoch {epochs:.2f}")

    bar = tqdm.tqdm(range(0, epochs), desc="Epochs", initial=0)
    for i in bar:
        var_params, loss, grad_norm, opt_state = opt_step(
            optimizer, opt_state, var_params, fix_params, b_dydt, b_ys
        )
        loss_val = float(jnp.squeeze(loss))
        grad_norm_val = float(jnp.squeeze(grad_norm))
        bar.set_postfix({"loss": f"{loss_val:.4e}", "gnorm": f"{grad_norm_val:.4e}"})

        if cfg["save_dir"]:  # checkpoint + scalar logging
            state = {"var_params": var_params, "opt_state": opt_state}
            mngr.standard_save(ckpt_mngr, i, state, loss)
            loss_logger.log(i, loss_val)
            grad_norm_logger.log(i, grad_norm_val)

        if cfg["save_dir"] and i % cfg["test_interval"] == 0:  # Testing
            async_worker.submit(_plot_k, var_params, f"logs/est_k_{i}.pdf")
            loss_logger.flush()
            grad_norm_logger.flush()


if cfg["infer"]:
    assert cfg["save_dir"], "must provide save_dir to load checkpoint"
    restore_step = ckpt_mngr.best_step()
    print(f"Inference using checkpoint {cfg['save_dir']}/checkpoints/{restore_step}")
    abstract_state = {"var_params": var_params, "opt_state": opt_state}
    state = mngr.standard_restore(ckpt_mngr, restore_step, abstract_state)

    _plot_k(state["var_params"], "est_k.pdf")

    combined_pred = data.combine_static_ro2(jax.tree.map(jnp.exp, state["var_params"]))
    np.savez(
        cfg["save_dir"] / "est_k.npz",
        truth=combined_true,
        pred=combined_pred,
        init=combined_init,
    )

if cfg["save_dir"]:
    print("Finalising")
    loss_logger.close()
    grad_norm_logger.close()
    ckpt_mngr.wait_until_finished()
    ckpt_mngr.close()
    async_worker.shutdown()
