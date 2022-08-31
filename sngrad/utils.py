"""Utilities for data manipulation."""
import jax.numpy as jnp

import torch
import random
import numpy

def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


def set_random_seeds(seed: int = 0) -> None:
    """Fixes random seeds to preserve reproducibility.
    
    See also: https://pytorch.org/docs/stable/notes/randomness.html
    """
    numpy.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)


def set_worker_seed(worker_id):
    """Sets data worker seeds to preserve reproducibility.

    See also: https://pytorch.org/docs/stable/notes/randomness.html
    """
    seed = torch.initial_seed() % 2**32
    numpy.random.seed(seed)
    random.seed(seed)


def set_generator_seed():
    """Sets generator seed to preserve reproducibility.

    See also: https://pytorch.org/docs/stable/notes/randomness.html
    """
    generator = torch.Generator()
    generator.manual_seed(0)
    return generator
