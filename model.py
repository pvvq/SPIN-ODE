import jax
import jax.numpy as jnp
import diffrax as dfx

import chemistry as ch


def forward(params: dict, inputs: dict):
    """
    Move the model state forward in time

    Model entrance, integrate time derivative of model state in time

    Args:
        params: model parameters
        inputs: model inputs
    Returns:
        integrated model state
    """

    def ode(t, y, args):
        """time derivative of model state"""
        dy_dt = ch.power_rate_law(y, args['k'], args["stoichiometry"])
        return dy_dt

    ts = inputs['ts']
    y0 = inputs['y0']
    solver_cfg = params['solver']

    sol = dfx.diffeqsolve(
        dfx.ODETerm(ode),
        dfx.Kvaerno5(),
        t0=ts[0],
        t1=ts[-1],
        y0=y0,
        saveat=dfx.SaveAt(ts=ts),
        dt0=None,
        max_steps=solver_cfg['max_steps'],
        stepsize_controller=dfx.PIDController(
            rtol=solver_cfg['rtol'], atol=solver_cfg['atol']
        ),
        throw=True,
        args=params,
    )
    return sol.ys

if __name__ == "__main__":
    # Example reaction:
    # NO+O3 -> NO2: 0.266 * 10^2
    stoi_reac = jnp.asarray([[1,1,0]]).T  # reaction stoichiometric matrix
    stoi_prod = jnp.asarray([[0,0,1]]).T  # product stoichiometric matrix
    K = jnp.asarray([0.266e2])            # reaction rate coefficient
    ts = jnp.arange(0, 1, 0.1)            # time span
    y0 = jnp.asarray([0.2, 0.04, 0])      # initial concentration

    params = {
        'k': K,
        'stoichiometry': ch.Stoichiometry(stoi_reac, stoi_prod),
        'solver': {
            'rtol': 1e-6,
            'atol': 1e-7,
            'max_steps': 8192,
        },
    }
    inputs = {
        'ts': ts,
        'y0': y0,
    }

    traj_measure = forward(params, inputs)
    print(traj_measure)


    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(6,2), layout='constrained')
    for i in range(3):
        axes[i].plot(inputs['ts'], traj_measure[:,i])
    fig.savefig("plots/demo_traj.png", dpi=300)