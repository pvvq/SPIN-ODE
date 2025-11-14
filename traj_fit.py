from pathlib import Path

import jax
import jax.numpy as jnp
import jaxtyping
import equinox as eqx
import optax
import tqdm
from torch.utils.data import DataLoader

import chem_data as cd
import model
import network as nn
import train_utils

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

# Model ========================================================================

neural_network = nn.ScaleMLP(
    sch.num_spc, scale,
    width_size=128, depth=3, key=jax.random.PRNGKey(cfg['seed'])
)

def mse(pred, target):
    return jnp.mean((pred - target) ** 2)

def scale_mse(pred, target):
    scale = jnp.max(target, axis=-2, keepdims=True)
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
    return scale_mse(traj_pred, batch['conc'])

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
import orbax.checkpoint as ocp
checkpointer = ocp.StandardCheckpointer()
ckpt_path = ocp.test_utils.erase_and_create_empty(Path('checkpoints/').absolute())
ckpt_cnt = 0

optimizer = optax.adam(cfg['learning_rate'])
opt_state = optimizer.init(eqx.filter(neural_network, eqx.is_inexact_array))

for first_n in jnp.arange(0.1, 1.1, 0.1):
    print("ratio of trajectory for training: ", first_n)
    traj_dataset = cd.TrajDataset(
        y_arr[:, :int(y_arr.shape[1]*first_n), :],
        t_arr[:, :int(t_arr.shape[1]*first_n)]
    )
    traj_dataloader = DataLoader(traj_dataset, batch_size=8, shuffle=True, collate_fn=cd.jax_collate)

    bar = tqdm.tqdm(range(0, cfg['epochs']), desc=f"Epochs", initial=0)
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
    saveable, _ = eqx.partition(neural_network, eqx.is_array_like)
    checkpointer.save(ckpt_path / f"{ckpt_cnt}", saveable)
    ckpt_cnt += 1