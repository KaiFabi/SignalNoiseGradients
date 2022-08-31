"""Prototype of optimization with signal-to-noise gradients (SNG).

Very basic implementation of signal-to-noise gradients. 
Implementation of sngrads is far from optimized.
Consider this implementation as a proof of concept.

Some basic parts of the dataloader and network of this code are from the following JAX tutorial:
https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
"""
import os
import time
import numpy as np

import jax 
from torch.utils.tensorboard import SummaryWriter

from sngrad.utils import one_hot, set_random_seeds
from sngrad.dataloader import DataServer
from sngrad.model import Model
from sngrad.lr_search import learning_rate_search


def run_experiment(optimizer: str):
    """Method runs experiments to compare standard gradients
    with signal-to-noise gradients."""

    set_random_seeds(seed=69)

    hparams = {
        # dataset options: mnist, fashion_mnist, cifar10
        "dataset": "fashion_mnist",
        "layer_sizes": [784, 512, 512, 512, 10],
        "step_size": -1,
        "num_epochs": 100,
        "batch_size": 512,
        "num_targets": 10,
        "num_workers": 4,
        "stats_every_num_epochs": 5,
        # optimizer options: sgd, sng
        "optimizer": optimizer,     
        # device options: tpu, gpu, cpu
        "device": "tpu",        
    }

    file = open(f"{hparams['optimizer']}_training.txt", "a")

    if hparams["device"] == "cpu":
        jax.config.update('jax_platform_name', 'cpu')

    if hparams["step_size"] == -1:
        best_lr = learning_rate_search(
            hparams=hparams,
            lr_min=1e-3,
            lr_max=1e+1,
            num_steps=100, 
            num_epochs=1, 
            )
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

    for epoch in range(num_epochs):

        start_time = time.time()

        for x, y in training_generator:
            y = one_hot(y, num_targets)
            model.step(x, y, step_size)

        epoch_time = time.time() - start_time

        if epoch % stats_every_num_epochs == 0:
            train_accuracy = model.accuracy(train_images, train_labels)
            test_accuracy = model.accuracy(test_images, test_labels)
            train_loss = model.loss(train_images, train_labels)
            test_loss = model.loss(test_images, test_labels)
            message = f"{epoch} {epoch_time:0.2f} {train_loss} {test_loss} {train_accuracy} {test_accuracy}"
            print(message)
            file.write(f"{message}\n")
            file.flush()
            os.fsync()

        writer.add_scalar("Epoch_time", epoch_time, epoch)
        writer.add_scalar("Accuracy/train", np.array(train_accuracy), epoch)
        writer.add_scalar("Accuracy/test", np.array(test_accuracy), epoch)
        writer.add_scalar("Loss/train", np.array(train_loss), epoch)
        writer.add_scalar("Loss/test", np.array(test_loss), epoch)

    writer.close()
    file.close()


if __name__ == "__main__":

    optimizers = ("sgd", "sng")

    for optimizer in optimizers:
        run_experiment(optimizer=optimizer)
