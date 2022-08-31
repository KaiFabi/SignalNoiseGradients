"""Minimal example to test if gradients are identical.
"""
import jax
import jax.numpy as jnp
from jax import random


def loss(params, x, t):
    w, b = params
    x = jnp.atleast_2d(x)
    t = jnp.atleast_2d(t)
    y = jnp.dot(w, x.T) + b
    return ((t.T * y)**2).sum()


def main():

    n_samples = 16
    dims_in = 32
    dims_out = 8

    key = random.PRNGKey(0)

    # Random data
    x = random.normal(key, (n_samples, dims_in), dtype=jnp.float32)
    t = random.normal(key, (n_samples, dims_out), dtype=jnp.float32)
    
    # Random weights
    w = random.normal(key, (dims_out, dims_in), dtype=jnp.float32)
    b = random.normal(key, (dims_out, 1), dtype=jnp.float32)
    params = (w, b)

    # Reduced gradients
    reduced_grads = jax.grad(loss, argnums=0)
    dw0, db0 = reduced_grads(params, x, t)

    # Per-example gradients
    perex_grads = jax.vmap(jax.grad(loss, argnums=0), in_axes=((None, None), 0, 0))
    dw1, db1 = perex_grads(params, x, t)

    print(jnp.allclose(dw0, jnp.sum(dw1, axis=0)))
    print(jnp.allclose(db0, jnp.sum(db1, axis=0)))
    

if __name__ == "__main__":
    main()