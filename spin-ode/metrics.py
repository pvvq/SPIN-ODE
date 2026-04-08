import jax
import jax.numpy as jnp
import jaxtyping


def mse(pred, target):
    return jnp.mean((pred - target) ** 2)

def scale_mse(pred, target, scale):
    return jnp.mean(((pred - target) / scale) ** 2)

def SMSPELoss(pred, truth):
    """Symmetric Mean Squared Percentage Error"""
    diff = jnp.sqrt(jnp.square(pred - truth))
    scale = jnp.sqrt(jnp.square(pred)) + jnp.sqrt(jnp.square(truth))
    return jnp.log(jnp.mean(diff / scale))

def signed_softlog(x, alpha=10.0):
    return jnp.sign(x) * jnp.log1p(alpha * jnp.abs(x))

def signed_softlog_mse(y_true, y_pred, alpha=10.0):
    y_true_slog = signed_softlog(y_true, alpha)
    y_pred_slog = signed_softlog(y_pred, alpha)
    return jnp.mean((y_pred_slog - y_true_slog) ** 2)

def LogMAELoss(pred, truth):
    return jnp.mean(jnp.abs(jnp.log(pred) - jnp.log(truth)))

def LogMSELoss(pred, truth):
    return jnp.mean(jnp.square(jnp.log(pred) - jnp.log(truth)))

def ScaleMSELoss(pred, truth, yscale):
    scaled_diff = (pred - truth) / yscale
    return jnp.mean(jnp.square(scaled_diff))

def ScaleMAELoss(pred, truth, yscale):
    scaled_diff = (pred - truth) / yscale
    return jnp.mean(jnp.abs(scaled_diff))

def TVLoss1D(x, scale, window_size=1):
    diff = x[window_size:, ...] - x[:-window_size, ...]
    tv = jnp.square(diff)

    loss = jnp.mean(tv / scale)
    return loss
