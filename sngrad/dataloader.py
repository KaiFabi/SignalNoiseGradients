"""Simple PyTorch-based dataloader for JAX.
"""
import numpy as np
from pathlib import Path

import jax.numpy as jnp

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, FashionMNIST, MNIST
from torchvision import transforms

from sngrad.utils import one_hot, set_generator_seed, set_worker_seed


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class NormFlattenCast(object):
    def __call__(self, data):
        return np.ravel(np.array(data, dtype=jnp.float32))


class DataServer:

    def __init__(self, hparams: dict) -> None:

        self.dataset = hparams["dataset"]
        self.batch_size =  hparams["batch_size"]
        self.num_targets = hparams["num_targets"]
        self.num_workers = hparams["num_workers"]

        home_dir = Path.home()
        root_dir = f"{home_dir}/data/{self.dataset}"
        # root_dir = "/tmp/" + f"{dataset}"

        print(self.dataset)
        if self.dataset == "cifar10":

            cifar10 = CIFAR10(root=root_dir, train=True, download=True)
            mean = jnp.mean(jnp.array(cifar10.data / 255.0, dtype=jnp.float32), axis=(0, 1, 2))
            std = jnp.std(jnp.array(cifar10.data / 255.0, dtype=jnp.float32), axis=(0, 1, 2))

            train_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                NormFlattenCast()
                ])

            test_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                NormFlattenCast(),
            ])

            self.train_dataset = CIFAR10(root=root_dir, train=True, download=True, transform=train_transforms)
            self.test_dataset = CIFAR10(root=root_dir, train=False, download=True, transform=test_transforms)

        elif self.dataset == "fashion_mnist":

            fashion_mnist = FashionMNIST(root=root_dir, train=True, download=True)
            mean = float(jnp.array(fashion_mnist.data / 255.0, dtype=jnp.float32).mean())
            std = float(jnp.array(fashion_mnist.data / 255.0, dtype=jnp.float32).std())

            train_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomErasing(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=20),
                transforms.Normalize(mean=mean, std=std),
                NormFlattenCast()
                ])

            test_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                NormFlattenCast(),
            ])

            self.train_dataset = FashionMNIST(root=root_dir, train=True, download=True, transform=train_transforms)
            self.test_dataset = FashionMNIST(root=root_dir, train=False, download=True, transform=test_transforms)

        elif self.dataset == "mnist":

            mnist = MNIST(root=root_dir, train=True, download=True)
            mean = float(jnp.array(mnist.data / 255.0, dtype=jnp.float32).mean())
            std = float(jnp.array(mnist.data / 255.0, dtype=jnp.float32).std())

            train_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomRotation(degrees=20),
                transforms.RandomErasing(),
                transforms.Normalize(mean=mean, std=std),
                NormFlattenCast()
                ])

            test_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                NormFlattenCast(),
            ])

            self.train_dataset = MNIST(root=root_dir, train=True, download=True, transform=train_transforms)
            self.test_dataset = MNIST(root=root_dir, train=False, download=True, transform=test_transforms)

        else:
            raise NotImplementedError(f"Dataset {self.dataset} not available.")

        self.generator = set_generator_seed()

    def get_training_dataloader(self):
        """Returns training data generator."""
        training_dataloader = DataLoader(
            dataset=self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            collate_fn=numpy_collate,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=set_worker_seed,
            generator=self.generator,
        )
        return training_dataloader

    def get_test_dataloader(self):
        """Returns test data generator."""
        test_dataloader = DataLoader(
            dataset=self.test_dataset, 
            batch_size=2 * self.batch_size, 
            num_workers=self.num_workers,
            collate_fn=numpy_collate,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=set_worker_seed,
            generator=self.generator,
        )
        return test_dataloader

    # def get_dataset(self):
    #     """Returns full training and test dataset."""
    #     # Get the full train dataset to compute accuray
    #     train_images = np.array(self.train_dataset.data).reshape(len(self.train_dataset.data), -1)
    #     train_labels = one_hot(np.array(self.train_dataset.targets), self.num_targets)
    #     # Get the full test dataset to compute accuray
    #     test_images = np.array(self.test_dataset.data).reshape(len(self.test_dataset.data), -1)
    #     test_labels = one_hot(np.array(self.test_dataset.targets), self.num_targets)
    #     return train_images, train_labels, test_images, test_labels
