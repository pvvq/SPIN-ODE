"""
Neural ODE fit trajectory

Usage: python traj_fit.py -h
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

jax.config.update("jax_enable_x64", True)
FTYPE = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

import model
import data
import manager as mngr
from metrics import scale_mse
import plot

cfg = mngr.load_config()
pprint(cfg)

key = jax.random.PRNGKey(cfg["seed"])

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
    "nn": neural_network,
}
fix_params = {"solver": cfg["solver"], "scale": scale}


def pred_ys(params, ts, y0):
    return model.solve(params, ts, y0, model.neural_ode)


def loss_fn(var_params, fix_params, ts, b_ys):
    params = {**var_params, **fix_params}
    b_ys_pred = eqx.filter_vmap(pred_ys, in_axes=(None, None, 0))(
        params, ts, b_ys[:, 0]
    )
    return scale_mse(b_ys_pred, b_ys, params["scale"]["yScale"])


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
    grad_norm = global_norm(eqx.filter(grads, eqx.is_inexact_array))
    updates, opt_state = optim.update(grads, opt_state, var_params)
    new_var_params = eqx.apply_updates(var_params, updates)
    return new_var_params, loss, grad_norm, opt_state


def _plot_ys(ys_pred, fname):
    fig = plot.plot_series(y=ys_pred, t=ts, yy=b_ys[0], tt=ts)
    fig.savefig(cfg["save_dir"] / fname)
    plot.plt.close(fig)


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

if cfg["resume"]:
    if cfg["resume_ckpt"]:
        restore_path = Path(cfg["resume_ckpt"]).absolute()
    else:
        assert cfg["save_dir"], "must provide save_dir or resume_ckpt to resume"
        restore_path = cfg["save_dir"] / "checkpoints"
    restore_mngr = mngr.get_checkpoint_manager(restore_path)
    start_epoch = restore_mngr.latest_step()
    print(f"Resume from checkpoint {restore_path}/{start_epoch}")

    arr_params, static_params = eqx.partition(var_params, eqx.is_array_like)
    abstract_state = {"var_params": arr_params, "opt_state": opt_state}
    restored = mngr.standard_restore(restore_mngr, start_epoch, abstract_state)
    var_params = eqx.combine(restored["var_params"], static_params)
    opt_state = restored["opt_state"]
else:
    start_epoch = 0


for length, epochs in zip(length_strategy, epochs_strategy):
    print(f"strategy: length {length:.2f}, epoch {epochs:.2f}")

    bar = tqdm.tqdm(range(start_epoch + 1, epochs + 1), desc="Epochs", ncols=120)
    for i in bar:
        var_params, loss, grad_norm, opt_state = opt_step(
            optimizer, opt_state, var_params, fix_params, ts, b_ys
        )
        loss_val = float(jnp.squeeze(loss))
        grad_norm_val = float(jnp.squeeze(grad_norm))
        bar.set_postfix(
            {
                "loss": f"{loss_val:.4e}",
                "gnorm": f"{grad_norm_val:.4e}",
            }
        )

        if cfg["save_dir"]:  # checkpoint + scalar logging
            arr_params, _ = eqx.partition(var_params, eqx.is_array_like)
            state = {"var_params": arr_params, "opt_state": opt_state}
            mngr.standard_save(ckpt_mngr, i, state, loss)
            loss_logger.log(i, loss_val)
            grad_norm_logger.log(i, grad_norm_val)

        if cfg["save_dir"] and i % cfg["test_interval"] == 0:  # Testing
            ys_pred = pred_ys({**var_params, **fix_params}, ts, b_ys[0, 0])
            async_worker.submit(_plot_ys, ys_pred, f"logs/traj_fit_{i}.pdf")

            loss_logger.flush()
            loss_logger.plot()
            grad_norm_logger.flush()
            grad_norm_logger.plot()


if cfg["infer"]:
    assert cfg["save_dir"], "must provide save_dir to load checkpoint"
    # load checkpoint
    restore_step = ckpt_mngr.best_step()
    print(
        f"Inference using best checkpoint {cfg['save_dir']}/checkpoints/{restore_step}"
    )
    arr_params, static_params = eqx.partition(var_params, eqx.is_array_like)
    restored = mngr.standard_restore(
        ckpt_mngr, restore_step, {"var_params": arr_params}
    )
    var_params = eqx.combine(restored["var_params"], static_params)

    ys_pred = pred_ys({**var_params, **fix_params}, ts, b_ys[0, 0])
    _plot_ys(ys_pred, "traj_fit.pdf")

if cfg["save_dir"]:
    print("Finalising")
    loss_logger.flush()
    grad_norm_logger.flush()
    ckpt_mngr.wait_until_finished()
    ckpt_mngr.close()
    async_worker.shutdown()
