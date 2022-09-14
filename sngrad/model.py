"""Simple network class for fully connected neural networks in JAX.

"""
from typing import List, Tuple

import jax.numpy as jnp
import jax

from jax.scipy.special import logsumexp
from jaxlib.xla_extension import DeviceArray


class Model:
    """Simple neural network class."""

    def __init__(self, hparams: dict):

        layer_sizes = hparams["layer_sizes"]
        optimizer = hparams["optimizer"]

        if optimizer == "sgd":
            self.update = _update_sgd
            self.backward = jax.jit(jax.grad(_loss, argnums=0))
        elif optimizer == "sng":
            self.update = _update_sng
            self.backward = jax.jit(jax.vmap(jax.grad(_loss, argnums=0), in_axes=(None, 0, 0)))

            # grads_type = "sng_v3"
            # if grads_type == "sng_v1":
            #     self._sng = _sng_v1
            # elif grads_type == "sng_v1":
            #     self._sng = _sng_v2
            # elif grads_type == "sng_v1":
            #     self._sng = _sng_v3
            # else:
            #     raise NotImplemented(f"Gradient computation {grads_type} not found.")

        else:
            raise NotImplemented(f"Optimizer {optimizer} not found.")

        self.params = self.init_network_params(layer_sizes, jax.random.PRNGKey(0))
        self.grads = None

    def init_network_params(self, sizes: list, key: DeviceArray):
        """Initialize all layers for a fully-connected neural network with sizes 'sizes'"""
        keys = jax.random.split(key, len(sizes))
        return [self._random_layer_params(fan_in, fan_out, key) 
                for fan_in, fan_out, key in zip(sizes[:-1], sizes[1:], keys)]

    @staticmethod
    def _random_layer_params(fan_in: int, fan_out: int, key: DeviceArray):
        """A helper function to randomly initialize weights and biases for a dense neural network layer"""
        w_key, _ = jax.random.split(key)
        scale = jnp.sqrt(2.0 / fan_in)
        return scale * jax.random.normal(w_key, (fan_out, fan_in)), jnp.zeros(shape=(fan_out, ))

    def step(self, x: DeviceArray, y: DeviceArray, step_size: float):
        self.grads = self.backward(self.params, x, y)
        self.params = self.update(self.params, self.grads, step_size)

    def loss_accuracy(self, images: DeviceArray, targets: DeviceArray):
        """Computes loss and accuracy."""
        images = jnp.atleast_2d(images)
        targets = jnp.atleast_2d(targets)

        preds = forward(self.params, images)

        target_class = jnp.argmax(targets, axis=1)
        predicted_class = jnp.argmax(preds, axis=1)

        accuracy = float(jnp.sum(predicted_class == target_class))
        loss = -float(jnp.sum(preds * targets))

        return loss, accuracy


@jax.jit
def _update_sgd(params: List[Tuple[DeviceArray]], grads: List[Tuple[DeviceArray]], lr: float):
    """Stochastic gradient descent with accumulated gradients."""
    return [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, grads)]


@jax.jit
def _update_sng(params: List[Tuple[DeviceArray]], grads: List[Tuple[DeviceArray]], lr: float):
    """Stochastic gradient descent with noise-adjusted gradients."""
    return [(w - lr * _sng_v1(dw), b - lr * _sng_v1(db)) for (w, b), (dw, db) in zip(params, grads)]


@jax.jit
def _loss(params: List[Tuple[DeviceArray]], images: DeviceArray, targets: DeviceArray):
    images = jnp.atleast_2d(images)
    targets = jnp.atleast_2d(targets)
    preds = forward(params, images)
    return -jnp.mean(preds * targets)


def relu(x: DeviceArray):
    """Rectified Linear Unit activation function."""
    return jnp.maximum(0.0, x)


def predict(params: DeviceArray, image: DeviceArray):
    """Per-example forward method."""
    activations = image

    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b

    return logits - logsumexp(logits)


# Make a batched version of the "predict" function using "vmap".
forward = jax.jit(jax.vmap(predict, in_axes=(None, 0)))


@jax.jit
def _sng_v1(grads: DeviceArray, alpha: float = 1.0):
    """Performs uncertainty-based gradient adjustment.

    Computes noise adjusted gradients by dividing gradients
    by their standard deviation.

    Args:
        grads: Gradients associated with batch.
        alpha: Multiplier controls the size of gradients.
    
    Returns:
        Modified aggregated gradients.
    """
    # Compute mean and standard deviation of per-example gradients.
    grads_mean = jnp.mean(grads, axis=0)
    grads_std = jnp.std(grads, axis=0)
    # Compute signal-to-noise ratio gradients.
    return grads_mean / (1.0 + alpha * grads_std)


@jax.jit
def _sng_v2(grads: DeviceArray, alpha: float = 1.0):
    """Performs uncertainty-based gradient adjustment.

    Computes noise adjusted gradients by multiplying
    aggregated gradients by squared signal-to-noise
    ratio.

    Args:
        grads: Gradients associated with batch.
        alpha: Multiplier controls the size of gradients.
    
    Returns:
        Modified aggregated gradients.
    """
    # Compute mean and standard deviation of per-example gradients.
    grads_mean = jnp.mean(grads, axis=0)
    grads_var = jnp.var(grads, axis=0)
    # Compute signal-to-noise ratio gradients.
    return grads_mean**3 / (1.0 + alpha * grads_var)


@jax.jit
def _sng_v3(grads: DeviceArray, alpha: float = 1.0):
    """Performs uncertainty-based gradient adjustment.

    Computes noise adjusted gradients by multiplying
    aggregated gradients by (1 - clipped(variance)).

    Aggregated gradients with no variance are being
    multiplied by 1. High aggregated gradients
    associated with high variance are multiplied by
    value in the range between [0, 1).

    Parameter alpha allows to stronger penalize variance.

    Args:
        grads: Gradients associated with batch.
        alpha: Multiplier controls the size of gradients.
    
    Returns:
        Modified aggregated gradients.
    """
    # Compute mean and standard deviation of per-example gradients.
    grads_mean = jnp.mean(grads, axis=0)
    grads_var = jnp.var(grads, axis=0)
    # Compute signal-to-noise ratio gradients.
    return (1.0 - jnp.clip(alpha * grads_var, a_max=1.0)) * grads_mean