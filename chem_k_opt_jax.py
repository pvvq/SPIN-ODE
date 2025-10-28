# Chemistry reaction rate coefficient optimisation using Jax
# 
# Copyright (c) 2025 Wenqing Peng (wenqing.peng@helsinki.fi)  


import jax
import jax.numpy as jnp
import diffrax as dfx
import tqdm
import matplotlib.pyplot as plt


# Example reaction:
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
    'stoi_reac': stoi_reac,
    'stoi_prod': stoi_prod,
}
sim_cfg = {
    'ts': ts,
    'y0': y0,
}

def forward(params, sim_cfg):
    def ode(t, y, args):
        """reaction rate law"""
        k = args['k']
        stoi_reac, stoi_prod = args['stoi_reac'], args['stoi_prod']
        
        rate = k * jnp.prod(y[:, None] ** stoi_reac, axis=0)
        dy_dt  = (stoi_prod - stoi_reac) @ rate
        
        return dy_dt

    ts = sim_cfg['ts']
    y0 = sim_cfg['y0']

    sol = dfx.diffeqsolve(
            dfx.ODETerm(ode),
            dfx.Kvaerno5(),
            t0=ts[0],
            t1=ts[-1],
            y0=y0,
            saveat=dfx.SaveAt(ts=ts),
            dt0=None,
            max_steps=8192,
            stepsize_controller=dfx.PIDController(rtol=1e-6, atol=1e-7),
            throw=True,
            args=params,
        )
    return sol.ys


params = {**est_params, **fix_params}
traj_measure = forward(params, sim_cfg)
print(traj_measure)


fig, axes = plt.subplots(1, 3, figsize=(6,2), layout='constrained')
for i in range(3):
    axes[i].plot(sim_cfg['ts'], traj_measure[:,i])
fig.savefig("plots/demo_traj.png", dpi=300)


# ## Automatic differentiation

def mse(prediction, target):
    return jnp.mean((prediction - target) ** 2)

@jax.jit
def loss_fn(estimated_params, fixed_params, sim_cfg, traj_measure):
    params = {**estimated_params, **fixed_params}
    traj_pred = forward(params, sim_cfg)
    return mse(traj_pred, traj_measure)


print("loss", loss_fn(est_params, fix_params, sim_cfg, traj_measure))


# - `jax.grad` transform a function so that it calculate the gradient w.r.t. (by default the 1st) input argument
# - pure jax function is autodiffed by applying the chain rule
# - `diffrax` library implements adjoint method to compute gradient of solution w.r.t. input and parameter effeciently

grad_fn = jax.grad(loss_fn)


print("grads", grad_fn(est_params, fix_params, sim_cfg, traj_measure))


# ## Automatic vectorisation

# ### Mannual vectorisation




ks = jnp.logspace(jnp.log10(K*0.1), jnp.log10(K*10), 20)
batched_est_params = jax.tree.map(
    lambda leaf: jnp.stack([leaf] * len(ks)),
    est_params,
)
batched_est_params['k'] = ks

# ### `jax.vmap`
# - `jax.vmap` transformed a function so that it takes batched input.
# - `in_axes` specifies the which batched dimension is added to which input.

loss_vfn = jax.jit(
    jax.vmap(
        loss_fn,
        in_axes=(0,None,None,None),
    )
)
grad_vfn = jax.jit(
    jax.vmap(
        grad_fn,
        in_axes=(0,None,None,None),
    )
)

L = loss_vfn(batched_est_params, fix_params, sim_cfg, traj_measure)
grads = grad_vfn(batched_est_params, fix_params, sim_cfg, traj_measure)

for kk, ll, dd in zip(ks, L, grads['k']):
    print(kk, ll, dd)

fig, ax = plt.subplots(1)
ax.plot(ks, L)
ax.set_xscale("log")
ax.set_title("L")
fig.savefig("plots/demo_L.png", dpi=300)

fig, ax = plt.subplots(1)
ax.plot(ks, jnp.squeeze(grads['k']))
ax.set_xscale("log")
ax.set_title("dL_dk")
fig.savefig("plots/demo_dL_dk.png", dpi=300)


# ## Gradient-descent optimisation

est_params['k'] = K * 0.1
learning_rate = 1e5
epoch = 1000

print(f"ground truth: K={K}")
print(f"before optimisation: {est_params}")
bar = tqdm.tqdm(range(0, epoch), desc=f"Epochs", initial=0)

for i in bar:
    grads = grad_fn(est_params, fix_params, sim_cfg, traj_measure)
    est_params['k'] = est_params['k'] - grads['k'] * learning_rate
    bar.set_postfix({'k0': f"{float(jnp.squeeze(est_params['k'])):.4e}"})

print(f"after optimisation: {est_params}")