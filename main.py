"""Prototype of optimization with signal-to-noise gradients (SNG).

Very basic implementation of signal-to-noise gradients. 
Implementation of sngrads is far from optimized.
Consider this implementation as a proof of concept.

Some basic parts of the dataloader and network of this code are from the following JAX tutorial:
https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
"""
import time
import json
import numpy as np

import jax 
from torch.utils.tensorboard import SummaryWriter

from sngrad.utils import one_hot, set_random_seeds, add_input_samples
from sngrad.dataloader import DataServer
from sngrad.model import Model
from sngrad.lr_search import learning_rate_search


def run_experiment(hparams: dict) -> None:
    """Method runs experiments to compare standard gradients
    with signal-to-noise gradients."""

    set_random_seeds(seed=69)

    file = open(f"{hparams['optimizer']}_training.txt", "w")

    if hparams["device"] == "cpu":
        jax.config.update('jax_platform_name', 'cpu')

    if hparams["step_size"] is None:
        best_lr = learning_rate_search(hparams=hparams)
        print(f"best_lr = {best_lr}")
        hparams["step_size"] = best_lr

    # Parameters
    step_size = hparams["step_size"]
    num_epochs = hparams["num_epochs"]
    num_targets = hparams["num_targets"]
    stats_every_num_epochs = hparams["stats_every_num_epochs"]

    data_server = DataServer(hparams=hparams)

    training_generator = data_server.get_generator()
    train_images, train_labels, test_images, test_labels = data_server.get_dataset()

    model = Model(hparams=hparams)
    writer = SummaryWriter(comment=f"_training_{hparams['optimizer']}")

    # add_input_samples(dataloader=training_generator, tag="test", writer=writer)

    for epoch in range(num_epochs):

        start_time = time.time()

        for x, y in training_generator:
            y = one_hot(y, num_targets)
            model.step(x, y, step_size)

        epoch_time = time.time() - start_time
        writer.add_scalar("Epoch_time", epoch_time, epoch)

        if epoch % stats_every_num_epochs == 0:

            train_accuracy = model.accuracy(train_images, train_labels)
            test_accuracy = model.accuracy(test_images, test_labels)
            train_loss = model.loss(train_images, train_labels)
            test_loss = model.loss(test_images, test_labels)

            message = f"{epoch} {epoch_time:0.2f} {train_loss} {test_loss} {train_accuracy} {test_accuracy}"
            print(message)
            file.write(f"{message}\n")
            file.flush()

            writer.add_scalar("Accuracy/train", np.array(train_accuracy), epoch)
            writer.add_scalar("Accuracy/test", np.array(test_accuracy), epoch)
            writer.add_scalar("Loss/train", np.array(train_loss), epoch)
            writer.add_scalar("Loss/test", np.array(test_loss), epoch)

    writer.close()
    file.close()


if __name__ == "__main__":

    hparams = {
        # dataset options: mnist, fashion_mnist, cifar10
        "dataset": "fashion_mnist",
        "layer_sizes": [28**2, 64, 64, 10],
        "lr_search": {
            "lr_min": None,
            "lr_max": None,
            "num_steps": 20, 
            "num_epochs": 1, 
        },
        "step_size": None,
        "num_epochs": 20,
        "batch_size": 512,
        "num_targets": 10,
        "num_workers": 4,
        "stats_every_num_epochs": 1,
        # optimizer options: sgd, sng
        "optimizer": None,     
        # device options: tpu, gpu, cpu
        "device": "tpu",        
    }
    print("Experiment SNG")
    hparams.update({"step_size": None})
    hparams.update({"step_size": 0.02})
    hparams.update({"optimizer": "sng"})
    hparams["lr_search"].update({"lr_min": 1e-3, "lr_max": 1e-1})
    print(json.dumps(hparams, indent=4, sort_keys=True))
    run_experiment(hparams=hparams)

    print("Experiment SGD")
    hparams.update({"step_size": None})
    hparams.update({"step_size": 0.1})
    hparams.update({"optimizer": "sgd"})
    hparams["lr_search"].update({"lr_min": 1e-2, "lr_max": 1e-0})
    print(json.dumps(hparams, indent=4, sort_keys=True))
    run_experiment(hparams=hparams)
