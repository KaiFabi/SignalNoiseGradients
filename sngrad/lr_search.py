"""Simple learning rate search."""
import time
import numpy as np

from torch.utils.tensorboard import SummaryWriter

import jax.numpy as jnp

from sngrad.dataloader import DataServer
from sngrad.model import Model
from sngrad.utils import one_hot


def learning_rate_search(
    hparams: dict,
    lr_min: float, 
    lr_max: float,
    num_steps: int,
    num_epochs: int,
    ) -> float:
    """Searches for best learning rate within a defined interval.

    Args:
        hparams: Hyperparameters.
        lr_min: Minimal learning rate.
        lr_max: Maximal learning rate.
        num_steps: Steps taken within interval.
        num_epochs: Training epochs for each step.
        comment: Option comment for Tensorboard files.

    Returns:
        Best learning rate.
    """

    # Parameters
    num_targets = hparams["num_targets"]
    learning_rates = np.array(jnp.geomspace(start=lr_min, stop=lr_max, num=num_steps))

    writer = SummaryWriter(comment=f"_lr_search_{hparams['optimizer']}")

    file = open(f"{hparams['optimizer']}_lr_search.txt", "a")

    # List to keep track of results
    hist_test_loss = list()

    for learning_rate in learning_rates:

        data_server = DataServer(hparams=hparams)

        training_generator = data_server.get_generator()
        train_images, train_labels, test_images, test_labels = data_server.get_dataset()

        model = Model(hparams=hparams)
        
        start_time = time.time()

        for _ in range(num_epochs):
            for x, y in training_generator:
                y = one_hot(y, num_targets)
                model.step(x, y, learning_rate)

        train_time = time.time() - start_time

        train_accuracy = model.accuracy(train_images, train_labels)
        test_accuracy = model.accuracy(test_images, test_labels)
        train_loss = model.loss(train_images, train_labels)
        test_loss = model.loss(test_images, test_labels)

        writer.add_hparams(
            hparam_dict={"lr": float(learning_rate)},
            metric_dict={
                "hparam/train_loss": train_loss,
                "hparam/test_loss": train_loss,
                "hparam/train_accuracy": train_accuracy, 
                "hparam/test_accuracy": test_accuracy,
                }
        )

        hist_test_loss.append(test_loss)

        message = f"{train_time:0.2f} {learning_rate} {train_loss} {test_loss} {train_accuracy} {test_accuracy}"
        print(message)
        file.write(f"{message}\n")

    writer.close()
    file.close()

    idx_best_lr = np.argmin(hist_test_loss)
    best_lr = learning_rates[idx_best_lr]

    return best_lr
