#!/usr/bin/env python
# coding: utf-8

# # Chemistry reaction rate coefficient optimisation using Jax
# 
# Copyright (c) 2025 Wenqing Peng (wenqing.peng@helsinki.fi)  
# Licensed under the MIT License. See <https://opensource.org/licenses/MIT> for details.

import jax
import jax.numpy as jnp
import diffrax as dfx
import tqdm
import matplotlib.pyplot as plt





# NO+O3 -> NO2: 0.266 * 10^2
stoi_reac=jnp.asarray([[1,1,0]]).T  # reaction stoichiometric matrix
stoi_prod=jnp.asarray([[0,0,1]]).T  # product stoichiometric matrix
K = jnp.asarray([0.266e2])          # reaction rate coefficient
ts = jnp.arange(0, 1, 0.1)          # time span
y0 = jnp.asarray([0.2, 0.04, 0])    # initial concentration


def solve(k, stoi_reac, stoi_prod, ts, y0):
    def ode(t, y, args):
        """reaction rate law"""
        k, stoi_reac, stoi_prod = args
      
        rate = k * jnp.prod(y[:, None] ** stoi_reac, axis=0)
        dy_dt  = (stoi_prod - stoi_reac) @ rate
      
        return dy_dt

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
            args=(k, stoi_reac, stoi_prod),
        )
    return sol.ys


traj_measure = solve(K, stoi_reac, stoi_prod, ts, y0)


print(traj_measure)


fig, axes = plt.subplots(1, 3, figsize=(6,2), layout='constrained')
for i in range(3):
    axes[i].plot(ts, traj_measure[:,i])
fig.savefig("plots/demo_traj.png", dpi=300)


# ## Automatic differentiation

def mse(prediction, target):
    return jnp.mean((prediction - target) ** 2)

@jax.jit
def loss_fn(k, stoi_reac, stoi_prod, ts, y0, traj_measure):
    traj_pred = solve(k, stoi_reac, stoi_prod, ts, y0)
    return mse(traj_pred, traj_measure)


loss_fn(K, stoi_reac, stoi_prod, ts, y0, traj_measure)


# - `jax.grad` transform a function so that it calculate the gradient w.r.t. (by default the 1st) input argument
# - pure jax function is autodiffed by applying the chain rule
# - `diffrax` library implements adjoint method to compute gradient of solution w.r.t. input and parameter effeciently

dL_dk_fn = jax.grad(loss_fn)


print(dL_dk_fn(K, stoi_reac, stoi_prod, ts, y0, traj_measure))


# ## Automatic vectorisation

# ### Mannual vectorisation




ks = jnp.logspace(jnp.log10(K*0.1), jnp.log10(K*10), 20)

L = [loss_fn(k, stoi_reac, stoi_prod, ts, y0, traj_measure) for k in ks]
dL_dk = [dL_dk_fn(k, stoi_reac, stoi_prod, ts, y0, traj_measure) for k in ks]

for kk, ll, dd in zip(ks, L, dL_dk):
    print(kk, ll, dd)


# ### `jax.vmap`
# - `jax.vmap` transformed a function so that it takes batched input.
# - `in_axes` specifies the which batched dimension is added to which input.




loss_vfn = jax.jit(
    jax.vmap(
        loss_fn,
        in_axes=(0,None,None,None,None,None),
    )
)
dL_dk_vfn = jax.jit(
    jax.vmap(
        dL_dk_fn,
        in_axes=(0,None,None,None,None,None),
    )
)





L = loss_vfn(ks, stoi_reac, stoi_prod, ts, y0, traj_measure)
dL_dk = dL_dk_vfn(ks, stoi_reac, stoi_prod, ts, y0, traj_measure)





fig, ax = plt.subplots(1)
ax.plot(ks, L)
ax.set_xscale("log")
ax.set_title("L")
fig.savefig("plots/demo_L.png", dpi=300)

fig, ax = plt.subplots(1)
ax.plot(ks, jnp.squeeze(dL_dk))
ax.set_xscale("log")
ax.set_title("dL_dk")
fig.savefig("plots/demo_dL_dk.png", dpi=300)


# ## Gradient-descent optimisation

k0 = K*0.1
learning_rate = 1e5
epoch = 1000

print(f"ground truth: K={K}")
print(f"before optimisation: k0={k0}")
bar = tqdm.tqdm(range(0, epoch), desc=f"Epochs", initial=0)

for i in bar:
    dL_dk = dL_dk_fn(k0, stoi_reac, stoi_prod, ts, y0, traj_measure)
    k0 = k0 - dL_dk * learning_rate
    bar.set_postfix({'k0': f"{float(jnp.squeeze(k0)):.4e}"})

print(f"after optimisation: k0={k0}")







