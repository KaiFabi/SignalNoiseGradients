"""Simple learning rate search."""
import time
import numpy as np

from torch.utils.tensorboard import SummaryWriter

import jax.numpy as jnp

from sngrad.dataloader import DataServer
from sngrad.model import Model
from sngrad.utils import one_hot, comp_loss_accuracy


def learning_rate_search(hparams: dict) -> float:
    """Searches for best learning rate within a defined interval.

    Args:
        hparams: Hyperparameters.

    Returns:
        Returns learning rate associated with lowest test loss.
    """

    # Parameters
    lr_min = hparams["lr_search"]["lr_min"]
    lr_max = hparams["lr_search"]["lr_max"]
    num_steps = hparams["lr_search"]["num_steps"]
    num_epochs = hparams["lr_search"]["num_epochs"]
    num_targets = hparams["num_targets"]

    learning_rates = np.array(jnp.geomspace(start=lr_min, stop=lr_max, num=num_steps))

    writer = SummaryWriter(comment=f"_lr_search_{hparams['optimizer']}")
    file = open(f"{hparams['optimizer']}_lr_search.txt", "w")

    hist_loss = list()

    for learning_rate in learning_rates:

        data_server = DataServer(hparams=hparams)

        training_generator = data_server.get_training_dataloader()
        test_generator = data_server.get_training_dataloader()

        model = Model(hparams=hparams)
        
        start_time = time.time()

        for _ in range(num_epochs):
            for x, y in training_generator:
                y = one_hot(y, num_targets)
                model.step(x, y, learning_rate)

        train_time = time.time() - start_time

        training_loss, training_accuracy = comp_loss_accuracy(model=model, data_generator=training_generator)
        test_loss, test_accuracy = comp_loss_accuracy(model=model, data_generator=test_generator)

        writer.add_hparams(
            hparam_dict={"lr": float(learning_rate)},
            metric_dict={
                "hparam/training_loss": training_loss,
                "hparam/training_accuracy": training_accuracy, 
                "hparam/test_loss": test_loss,
                "hparam/test_accuracy": test_accuracy,
                }
        )

        hist_loss.append(training_loss)

        message = f"{train_time:0.2f} {learning_rate:.6f} {training_loss:.4f} {test_loss:.4f} {training_accuracy:.4f} {test_accuracy:.4f}"
        file.write(f"{message}\n")
        file.flush()
        print(message)

    writer.close()
    file.close()

    idx_best_lr = np.nanargmin(hist_loss)
    best_lr = float(learning_rates[idx_best_lr])

    return best_lr
