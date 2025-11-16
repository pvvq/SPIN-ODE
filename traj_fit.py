from pathlib import Path

import jax
import jax.numpy as jnp
import jaxtyping
import equinox as eqx
import optax
import tqdm
from torch.utils.data import DataLoader
import orbax.checkpoint as ocp

import chem_data as cd
import model
import network as nn
import train_utils
import plots.plot as pp


cfg = train_utils.load_config()

# Data =========================================================================

sch = cd.ROBER()

y_arr, t_arr = sch.data(1)
scale = {
    'yMax' : jnp.max(y_arr, axis=(0,1)),
    'yMin' : jnp.min(y_arr, axis=(0,1)),
    'tScale' : t_arr[0,-1] - t_arr[0,0],  # same tscale for every traj
}
scale['yScale'] = jnp.where(scale['yMax']-scale['yMin'] == 0.0, scale['yMax'], scale['yMax']-scale['yMin'])
scale['ytScale'] = scale['yScale'] / scale['tScale']
print("scale", scale)

# Model ========================================================================

neural_network = nn.ScaleMLP(
    sch.num_spc, scale,
    width_size=128, depth=3, key=jax.random.PRNGKey(cfg['seed'])
)

def mse(pred, target):
    return jnp.mean((pred - target) ** 2)

def scale_mse(pred, target, *, dim):
    """Scale loss by the maximum in 'dim' dimension"""
    scale = jnp.max(target, axis=dim, keepdims=True)
    return jnp.mean(((pred - target) / scale) ** 2)

def loss_fn(neu_net, batch):
    params = {
        'neural_network': neu_net,
        'solver': {
            'rtol': cfg['solver']['rtol'],
            'atol': cfg['solver']['atol'],
            'max_steps': cfg['solver']['max_steps'],
        },
    }
    inputs = {
        'y0': batch['conc'][:,0,:],
        'ts': batch['time'],
    }
    traj_pred = eqx.filter_vmap(model.forward, in_axes=(None,0,None))(
        params, inputs, model.neural_ode
    )
    return scale_mse(traj_pred, batch['conc'], dim=-2)

@eqx.filter_jit
def opt_step(
        optim: optax.GradientTransformation, opt_state: jaxtyping.PyTree,
        neu_net, batch
    ):
    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(
        neu_net, batch
    )
    updates, opt_state = optim.update(grads, opt_state)
    new_neu_net = eqx.apply_updates(
        neu_net, updates
    )
    return new_neu_net, loss_val, opt_state

# Training =====================================================================
checkpointer = ocp.StandardCheckpointer()
ckpt_path = Path('checkpoints/').absolute()
# ckpt_path = ocp.test_utils.erase_and_create_empty(ckpt_path)
ckpt_idx = 2

# load checkpoint
state, static = eqx.partition(neural_network, eqx.is_array_like)
abstract_state = jax.tree.map(ocp.utils.to_shape_dtype_struct, state)
restored_state = checkpointer.restore(ckpt_path / f"{ckpt_idx}", abstract_state)
neural_network = eqx.combine(restored_state, static)

optimizer = optax.adam(cfg['learning_rate'])
opt_state = optimizer.init(eqx.filter(neural_network, eqx.is_inexact_array))

length_strategy = [(jnp.asarray(0), end) for end in jnp.arange(0.3, 1, 0.1)]
epochs_strategy = (jnp.ones(len(length_strategy)) * 0.5 * cfg['epochs']).astype(int)
for (start, end), epochs in zip(length_strategy, epochs_strategy):
    print(f"trajectory slice: {start:.2f} {end:.2f}")
    start_idx = (y_arr.shape[1] * start).astype(int)
    end_idx = (y_arr.shape[1] * end).astype(int)

    traj_dataset = cd.TrajDataset(
        y_arr[:, start_idx:end_idx, :],
        t_arr[:, start_idx:end_idx]
    )
    traj_dataloader = DataLoader(traj_dataset, batch_size=4, shuffle=False, collate_fn=cd.jax_collate)

    bar = tqdm.tqdm(range(0, epochs), desc=f"Epochs", initial=0)
    for i in bar:
        for batch in traj_dataloader:
            neural_network, loss_val, opt_state = opt_step(
                optimizer, opt_state,
                neural_network, batch
            )
        bar.set_postfix({
            'loss': f"{float(jnp.squeeze(loss_val)):.4e}",
        })

    # checkpoint
    model_state, _ = eqx.partition(neural_network, eqx.is_array_like)
    ckpt_idx += 1
    checkpointer.save(ckpt_path / f"{ckpt_idx}", model_state)

    # Testing
    params = {
        'neural_network': neural_network,
        'solver': {
            'rtol': cfg['solver']['rtol'],
            'atol': cfg['solver']['atol'],
            'max_steps': cfg['solver']['max_steps'],
        },
    }
    inputs = {
        'y0': y_arr[0, start_idx],
        'ts': t_arr[0, start_idx:end_idx],
    }
    traj_pred = model.forward(params, inputs, model.neural_ode)
    traj_true = y_arr[0, start_idx:end_idx]
    print(traj_pred)
    print(traj_true)
    fig = pp.plot_series(y=traj_pred, t=inputs['ts'], yy=traj_true, tt=inputs['ts'])
    fig.savefig(f"plots/traj_fit_{start_idx:.1f}-{end_idx:.1f}.png", dpi=300)