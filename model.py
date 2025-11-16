from typing import Callable

import jax
import jax.numpy as jnp
import diffrax as dfx

import chemistry as ch

def physical_ode(t, y, args):
    """time derivative of model state"""
    dy_dt = ch.power_rate_law(y, args['k'], args["stoichiometry"])
    return dy_dt

def neural_ode(t, y, args):
    nn = args['neural_network']
    return nn(t, y)

def forward(params: dict, inputs: dict, ode: Callable):
    """
    Move the model state forward in time

    Model entrance, integrate time derivative of model state in time

    Args:
        params: model parameters
        inputs: model inputs
        ode: function of time derivative
    Returns:
        integrated model state
    """
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
    import chem_data as cd

    sch = cd.POLLU()

    params = {
        'k': jnp.asarray(sch.rconst),
        'stoichiometry': ch.Stoichiometry(
            sch.stoi_reac, sch.stoi_prod, sch.RO2_IDX, sch.RO2_K_IDX
        ),
        'solver': {
            'rtol': 1e-6,
            'atol': 1e-7,
            'max_steps': 8192,
        },
    }
    inputs = {
        'ts': jnp.asarray(cd.pollu_t),
        'y0': jnp.asarray(cd.pollu_y0),
    }
    traj = forward(params, inputs, physical_ode)

    # from Verwer, 1994
    true_end_conc_text = """
    5.64625548e-02 1.34248413e-01 4.13973433e-09 5.52314021e-03
    2.01897726e-07 1.46454186e-07 7.78424912e-02 3.24507535e-01
    7.49401338e-03 1.62229316e-08 1.13586383e-08 2.23050598e-03
    2.08716288e-04 1.39692102e-05 8.96488486e-03 4.35284637e-18
    6.89921970e-03 1.00780304e-04 1.77214651e-06 5.68294329e-05
    """
    true_end_conc = jnp.fromstring(true_end_conc_text, sep=' ')

    print("Regression test: ", jnp.allclose(traj[-1], true_end_conc))