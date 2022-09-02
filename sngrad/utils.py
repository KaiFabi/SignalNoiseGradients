"""Utilities for data manipulation."""
import jax.numpy as jnp

import torch
import random
import numpy
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


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


def set_worker_seed(worker_id) -> None:
    """Sets data worker seeds to preserve reproducibility.

    See also: https://pytorch.org/docs/stable/notes/randomness.html
    """
    seed = torch.initial_seed() % 2**32
    numpy.random.seed(seed)
    random.seed(seed)


def set_generator_seed() -> torch.Generator:
    """Sets generator seed to preserve reproducibility.

    See also: https://pytorch.org/docs/stable/notes/randomness.html
    """
    generator = torch.Generator()
    generator.manual_seed(0)
    return generator

 
def add_input_samples(
    dataloader: DataLoader, 
    tag: str, 
    writer: SummaryWriter, 
    global_step: int = 0, 
    n_samples: int = 16) -> None:
   """Add samples from dataloader to Tensorboard.

   Check if the input to the model is as expected.
   Useful for debugging.

   Args:
       dataloader:
       tag:
       writer:
       global_step:
       n_samples:
   """
   x, _ = next(iter(dataloader))

   n_samples = min(len(x), n_samples)
   x = x[:n_samples]

   x_min = np.array(jnp.min(x))
   x_max = np.array(jnp.max(x))
   x = (x - x_min) / (x_max - x_min)

   input_shape = dataloader.dataset.data.shape[1:]
   x = x.reshape(-1, *input_shape)
   x = x.transpose(0, 3, 1, 2)

   x = torch.tensor(x)

   writer.add_images(tag=f"sample_batch_{tag}", img_tensor=x, global_step=global_step)
