# Usage: python traj_fit.py 
#           --config <config_path>
#           --target <yaml_target>
#           --log/--no-log

from pathlib import Path

import jax
import jax.numpy as jnp
import jaxtyping
import equinox as eqx
import optax
import tqdm
from torch.utils.data import DataLoader
import orbax.checkpoint as ocp
import numpy as np

import chem_data as cd
import model
import network as nn
import train_utils
import plots.plot as pp


jax.config.update("jax_enable_x64", True)
cfg = train_utils.load_config()

# Data =========================================================================

sch = cd.POLLU()

def MAD(x, axis=None):
    m = jnp.median(x, axis=axis, keepdims=True)
    return jnp.median(jnp.abs(x - m), axis=axis)

y_arr, t_arr = sch.data(1, t=np.linspace(0, 60, 20))
print(t_arr)
scale = {
    'yMax' : jnp.max(y_arr, axis=(0,1)),
    'yMin' : jnp.min(y_arr, axis=(0,1)),
    'tScale' : jnp.asarray(t_arr[0,-1] - t_arr[0,0]),  # same tscale for every traj
}
scale['yScale'] = jnp.where(scale['yMax']-scale['yMin'] == 0.0, scale['yMax'], scale['yMax']-scale['yMin'])
scale['ytScale'] = scale['yScale'] / scale['tScale']
scale['median'] = jnp.median(y_arr, axis=(0,1))
scale['deviation'] = 1.4826 * MAD(y_arr, axis=(0,1))
print(scale)

# Model ========================================================================

neural_network = nn.ScaleMLP(
    data_size=sch.num_spc, width_size=10, depth=4,
    key=jax.random.PRNGKey(cfg['seed']),
)

params = {
    'solver': cfg['solver'],
    'scale': scale
}

def mse(pred, target):
    return jnp.mean((pred - target) ** 2)

def scale_mse(pred, target, scale):
    return jnp.mean(((pred - target) / scale) ** 2)

def loss_fn(neu_net, params, batch):
    params['neural_network'] = neu_net
    inputs = {
        'y0': batch['conc'][:,0,:],
        'ts': batch['time'],
    }
    traj_pred = eqx.filter_vmap(model.forward, in_axes=(None,0,None))(
        params, inputs, model.neural_ode
    )
    return scale_mse(traj_pred, batch['conc'], params['scale']['yScale'])

@eqx.filter_jit
def opt_step(
        optim: optax.GradientTransformation, opt_state: jaxtyping.PyTree,
        neu_net, params, batch
    ):
    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(
        neu_net, params, batch
    )
    updates, opt_state = optim.update(grads, opt_state, neu_net)
    new_neu_net = eqx.apply_updates(
        neu_net, updates
    )
    return new_neu_net, loss_val, opt_state

# Training =====================================================================
checkpointer = ocp.StandardCheckpointer()
ckpt_path = Path('checkpoints/').absolute()
ckpt_path = ocp.test_utils.erase_and_create_empty(ckpt_path)
ckpt_idx = 0

# load checkpoint
# state, static = eqx.partition(neural_network, eqx.is_array_like)
# abstract_state = jax.tree.map(ocp.utils.to_shape_dtype_struct, state)
# restored_state = checkpointer.restore(ckpt_path / f"{ckpt_idx}", abstract_state)
# neural_network = eqx.combine(restored_state, static)

optimizer = optax.chain(
    optax.clip(10),
    optax.adamw(cfg['learning_rate']),
    # optax.contrib.reduce_on_plateau(factor=0.5, patience=100),
)
opt_state = optimizer.init(eqx.filter(neural_network, eqx.is_inexact_array))

length_strategy = [(jnp.asarray(0), end) for end in jnp.asarray([1.0])]
epochs_strategy = (jnp.ones(len(length_strategy)) * cfg['epochs']).astype(int)
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
                neural_network, params, batch
            )
        bar.set_postfix({
            'loss': f"{float(jnp.squeeze(loss_val)):.4e}",
        })

        if i % 500 == 0:
            # checkpoint
            model_state, _ = eqx.partition(neural_network, eqx.is_array_like)
            ckpt_idx += 1
            checkpointer.save(ckpt_path / f"{ckpt_idx}", model_state)

        if i % 100 == 0:
            # Testing
            inputs = {
                'y0': y_arr[0, start_idx],
                'ts': t_arr[0, start_idx:end_idx],
            }
            traj_pred = model.forward({**params, **{'neural_network': neural_network}}, inputs, model.neural_ode)
            traj_true = y_arr[0, start_idx:end_idx]
            print(traj_pred[-1])
            print(traj_true[-1])
            print(scale_mse(traj_pred, traj_true, params['scale']['yScale']))
            fig = pp.plot_series(y=traj_pred, t=inputs['ts'], yy=traj_true, tt=inputs['ts'])
            fig.savefig(f"plots/traj_fit_{start_idx:.1f}-{end_idx:.1f}.png", dpi=300)