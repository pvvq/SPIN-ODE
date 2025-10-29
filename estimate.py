import jax
import jax.numpy as jnp
import jaxtyping
import equinox as eqx
import optax
import tqdm

import chemistry as ch
import model
import chem_data as cm

sch = cm.POLLU()

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
init_conc = jnp.zeros(20)
init_conc = init_conc.at[jnp.asarray([1,3,6,7,8,16])].set([0.2, 0.04, 0.1, 0.3, 0.01, 0.007])
inputs = {
    'ts': jnp.linspace(0,0.1,10),
    'y0': init_conc,
}

y_ture = model.forward({**est_params, **fix_params}, inputs)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(4, 5, figsize=(10,8), layout='constrained')
for i in range(sch.num_spc):
    row, col = i // 5, i % 5
    axes[row][col].plot(inputs['ts'], y_ture[:,i])
fig.savefig("plots/pollu_traj.png", dpi=300)

def mse(prediction, target):
    return jnp.mean((prediction - target) ** 2)

def loss_fn(estimated_params, fixed_params, inputs, y_ture):
    params = {**estimated_params, **fixed_params}
    traj_pred = model.forward(params, inputs)
    return mse(traj_pred, y_ture)

@eqx.filter_jit
def opt_step(
        optim: optax.GradientTransformation, opt_state: jaxtyping.PyTree,
        estimated_params, fixed_params, inputs,
        y_ture
    ):
    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(
        estimated_params, fixed_params, inputs, y_ture
    )
    updates, opt_state = optim.update(grads, opt_state)
    new_estimated_params = eqx.apply_updates(
        estimated_params, updates
    )
    return new_estimated_params, loss_val, opt_state


print(f"ground truth: K={est_params['k']}")
est_params['k'][4] = est_params['k'][4] * 0.9
LEARNING_RATE = 1e1
EPOCHS = 100

optimizer = optax.sgd(LEARNING_RATE)
opt_state = optimizer.init(est_params)

print(f"before optimisation: {est_params}")
bar = tqdm.tqdm(range(0, EPOCHS), desc=f"Epochs", initial=0)

for i in bar:
    est_params, loss_val, opt_state = opt_step(
        optimizer, opt_state,
        est_params, fix_params, inputs,
        y_ture
    )
    bar.set_postfix({
        'loss': f"{float(jnp.squeeze(loss_val)):.4e}",
    })

print(f"after optimisation: {est_params}")