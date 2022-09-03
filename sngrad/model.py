"""Simple network class for fully connected neural networks in JAX.

"""
import math
import jax.numpy as jnp
import jax

from jax.scipy.special import logsumexp


class Model:
    """Simple neural network class."""

    def __init__(self, hparams: dict):

        layer_sizes = hparams["layer_sizes"]
        optimizer = hparams["optimizer"]

        if optimizer == "sgd":
            self.update = self.update_sgd
            self.backward = jax.jit(jax.grad(self._loss, argnums=0))
        elif optimizer == "sng":
            self.update = self.update_sng
            self.backward = jax.jit(jax.vmap(jax.grad(self._loss, argnums=0), in_axes=(None, 0, 0)))
        else:
            raise NotImplemented(f"Optimizer {optimizer} not found.")

        self.params = self.init_network_params(layer_sizes, jax.random.PRNGKey(0))

    def init_network_params(self, sizes, key):
        """Initialize all layers for a fully-connected neural network with sizes 'sizes'"""
        keys = jax.random.split(key, len(sizes))
        return [self.random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

    def random_layer_params(self, m, n, key):
        """A helper function to randomly initialize weights and biases for a dense neural network layer"""
        w_key, _ = jax.random.split(key)
        scale = math.sqrt(2.0 / m)
        return scale * jax.random.normal(w_key, (n, m)), jnp.zeros(shape=(n, ))

    def step(self, x, y, step_size):
        self.update(x, y, step_size)

    def loss_accuracy(self, images, targets):
        """Computes loss and accuracy."""
        images = jnp.atleast_2d(images)
        targets = jnp.atleast_2d(targets)

        preds = forward(self.params, images)

        target_class = jnp.argmax(targets, axis=1)
        predicted_class = jnp.argmax(preds, axis=1)

        accuracy = float(jnp.sum(predicted_class == target_class))
        loss = -float(jnp.sum(preds * targets))

        return loss, accuracy

    @staticmethod
    @jax.jit
    def _loss(params, images, targets):
        images = jnp.atleast_2d(images)
        targets = jnp.atleast_2d(targets)
        preds = forward(params, images)
        return -jnp.mean(preds * targets)

    def update_sgd(self, x, y, step_size):
        """Method implements standard stochastic gradient descent with accumulated gradients."""
        grads = self.backward(self.params, x, y)
        self.params = [(w - step_size * dw, b - step_size * db) for (w, b), (dw, db) in zip(self.params, grads)]

    def update_sng(self, x, y, step_size):
        """Method implements standard stochastic gradient descent with noise-adjusted gradients."""
        grads = self.backward(self.params, x, y)
        self.params = [(w - step_size * self._sngrad(dw), b - step_size * self._sngrad(db)) 
                       for (w, b), (dw, db) in zip(self.params, grads)]

    @staticmethod
    @jax.jit
    def _sngrad(dx, eps: float = 1e-05):
        """Performs uncertainty-based gradient adjustment.

        Computes noise adjusted gradients by dividing gradients
        by their standard deviation.
        """
        # Compute mean and standard deviation
        # of per-example gradients.
        dx_mean = jnp.mean(dx, axis=0)
        dx_std = jnp.std(dx, axis=0)
        # Compute signal-to-noise ratio gradients.
        return dx_mean / (dx_std + eps)

    @staticmethod
    @jax.jit
    def _sngrad_v2(dx, eps: float = 1e-05):
        """Performs uncertainty-based gradient adjustment.

        Computes noise adjusted gradients by multiplying
        aggregated gradients by squared signal-to-noise
        ratio.
        """
        # Compute mean and standard deviation
        # of per-example gradients.
        dx_mean = jnp.mean(dx, axis=0)
        dx_var = jnp.var(dx, axis=0)
        # Compute signal-to-noise ratio gradients.
        return dx_mean**3 / (dx_var + eps)


def relu(x):
    """Rectified Linear Unit activation function."""
    return jnp.maximum(0.0, x)


def predict(params, image):
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
