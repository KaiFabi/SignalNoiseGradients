"""Tests for correct gradients.

TODO: 
    - Add tests.

"""
import jax
import jax.numpy as jnp


# @jit
def update_compare_gradients(params, x, y, loss, step_size):
    """Checks if gradients of are equal."""

    #######################
    # Accumulated gradients
    #######################
    grads_0 = jax.grad(loss, argnums=0)(params, x, y)  # the original

    #######################
    # Gradients per example
    #######################
    grads_1 = jax.vmap(jax.grad(loss, argnums=0), in_axes=(None, 0, 0))(params, x, y) 

    for i, ((dw0, db0), (dw1, db1)) in enumerate(zip(grads_0, grads_1)):
        print(f"Layer {i + 1}")
        print(jnp.allclose(dw0, jnp.mean(dw1, axis=0)))
        print(jnp.allclose(db0, jnp.mean(db1, axis=0)))
        print()
