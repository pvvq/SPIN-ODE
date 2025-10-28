import jax
import jax.numpy as jnp
import equinox as eqx


# Variables ====================================================================
class Stoichiometry(eqx.Module):
    reac: jax.Array
    prod: jax.Array
    net: jax.Array
    RO2_IDX: jax.Array
    RO2_K_IDX: jax.Array

    def __init__(
            self,
            reac: jax.typing.ArrayLike,
            prod: jax.typing.ArrayLike,
            RO2_IDX: jax.typing.ArrayLike = jnp.empty(0),
            RO2_K_IDX: jax.typing.ArrayLike = jnp.empty(0),
        ):
        self.reac = jnp.asarray(reac)
        self.prod = jnp.asarray(prod)
        self.net = jnp.asarray(prod - reac)
        self.RO2_IDX = jnp.asarray(RO2_IDX, dtype=jnp.int32)
        self.RO2_K_IDX = jnp.asarray(RO2_K_IDX, dtype=jnp.int32)


# Reaction rate law ============================================================

def power_rate_law(
        y: jax.Array,
        k: jax.Array,
        stoi: Stoichiometry,
    ):
    """ Reaction rate law (power-multiply formula)

    Args:
        y: concentrations.
        k: rate coefficients.
        stoi: Stoichiometriy.
    """
    y = jnp.clip(y, 1e-30, 1e30)
    rate = k * jnp.prod(y[:, None] ** stoi.reac, axis=0)

    RO2 = jnp.sum(y[stoi.RO2_IDX])
    rate = rate.at[stoi.RO2_K_IDX].multiply(RO2, unique_indices=True)

    dy_dt  = stoi.net @ rate
    return dy_dt

def log_rate_law(
        y: jax.Array,
        k: jax.Array,
        stoi: Stoichiometry,
    ):
    """ Reaction rate law (log-plus formula)

    Args:
        y: concentrations.
        k: rate coefficients.
        stoi: Stoichiometriy.
    """
    log_y = jnp.log(jnp.clip(y, 1e-30, 1e30))
    rate = k * jnp.exp(log_y @ stoi.reac)
    
    RO2 = jnp.sum(y[stoi.RO2_IDX])
    rate = rate.at[stoi.RO2_K_IDX].multiply(RO2, unique_indices=True)

    dy_dt = stoi.net @ rate
    return dy_dt