import jax
import jax.numpy as jnp
import jaxtyping
import equinox as eqx
import optax
import tqdm

import chemistry as ch
import model


# NO+O3 -> NO2: 0.266 * 10^2
stoi_reac=jnp.asarray([[1,1,0]]).T  # reaction stoichiometric matrix
stoi_prod=jnp.asarray([[0,0,1]]).T  # product stoichiometric matrix
K = jnp.asarray([0.266e2])          # reaction rate coefficient
ts = jnp.arange(0, 1, 0.1)          # time span
y0 = jnp.asarray([0.2, 0.04, 0])    # initial concentration

est_params = {
    'k': K,
}
fix_params = {
    'stoichiometry': ch.Stoichiometry(stoi_reac, stoi_prod),
}
inputs = {
    'ts': ts,
    'y0': y0,
}

y_ture = model.forward({**est_params, **fix_params}, inputs)

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


est_params['k'] = K * 0.1
LEARNING_RATE = 1e5
EPOCHS = 1000

optimizer = optax.sgd(LEARNING_RATE)
opt_state = optimizer.init(est_params)

print(f"ground truth: K={K}")
print(f"before optimisation: {est_params}")
bar = tqdm.tqdm(range(0, EPOCHS), desc=f"Epochs", initial=0)

for i in bar:
    est_params, loss_val, opt_state = opt_step(
        optimizer, opt_state,
        est_params, fix_params, inputs,
        y_ture
    )
    bar.set_postfix({'k0': f"{float(jnp.squeeze(est_params['k'])):.4e}"})

print(f"after optimisation: {est_params}")