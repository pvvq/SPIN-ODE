import jax
import jax.numpy as jnp
import jaxtyping
import equinox as eqx
import optax
import tqdm
from torch.utils.data import DataLoader

import chem_data as cd
import chemistry as ch
import model

sch = cd.ROBER()

est_params = {
    'k': sch.rconst,
}
fix_params = {
    'stoichiometry': ch.Stoichiometry(
        sch.stoi_reac, sch.stoi_prod, sch.RO2_IDX, sch.RO2_K_IDX
    ),
    'solver': {
        'rtol': 1e-3,
        'atol': 1e-4,
        'max_steps': 8192,
    },
}

y_arr, t_arr = sch.data(1)
traj_dataset = cd.TrajDataset(y_arr, t_arr)
traj_dataloader = DataLoader(traj_dataset, batch_size=8, shuffle=True, collate_fn=cd.jax_collate)

def mse(pred, target):
    return jnp.mean((pred - target) ** 2)

def scale_mse(pred, target):
    scale = jnp.max(target, axis=-2, keepdims=True)
    return jnp.mean(((pred - target) / scale) ** 2)

def loss_fn(estimated_params, fixed_params, batch):
    params = {**estimated_params, **fixed_params}
    inputs = {
        'y0': batch['conc'][:,0,:],
        'ts': batch['time'],
    }
    traj_pred = eqx.filter_vmap(model.forward, in_axes=(None,0,None))(
        params, inputs, model.physical_ode
    )
    return scale_mse(traj_pred, batch['conc'])

@eqx.filter_jit
def opt_step(
        optim: optax.GradientTransformation, opt_state: jaxtyping.PyTree,
        estimated_params, fixed_params, batch
    ):
    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(
        estimated_params, fixed_params, batch
    )
    updates, opt_state = optim.update(grads, opt_state)
    new_estimated_params = eqx.apply_updates(
        estimated_params, updates
    )
    return new_estimated_params, loss_val, opt_state


print(f"ground truth: K={est_params['k']}")
est_params['k'] = est_params['k'] * 0.99
print(f"before optimisation: {est_params}")

LEARNING_RATE = 1e-3
EPOCHS = 100

optimizer = optax.sgd(LEARNING_RATE)
opt_state = optimizer.init(est_params)

bar = tqdm.tqdm(range(0, EPOCHS), desc=f"Epochs", initial=0)
for i in bar:
    for batch in traj_dataloader:
        est_params, loss_val, opt_state = opt_step(
            optimizer, opt_state,
            est_params, fix_params, batch
        )
    bar.set_postfix({
        'loss': f"{float(jnp.squeeze(loss_val)):.4e}",
    })

print(f"after optimisation: {est_params}")