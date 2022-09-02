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


class NumpyLoader(DataLoader):
    def __init__(
        self, 
        dataset, 
        batch_size=1,
        shuffle=True, 
        sampler=None,
        batch_sampler=None, 
        num_workers=0,
        pin_memory=True, 
        drop_last=True,
        timeout=0, 
        worker_init_fn=None,
        generator=None
        ):

        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            generator=generator
        )


class NormFlattenCast(object):
    def __call__(self, data):
        data = 2.0 * (np.array(data, dtype=jnp.float32) / 255.0) - 1.0
        return np.ravel(data)


train_transforms = transforms.Compose([
    transforms.ToTensor(),
    # transforms.RandomRotation(degrees=20),
    transforms.RandomErasing(),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    transforms.ToPILImage(),
    NormFlattenCast()
    ])

test_transforms = transforms.Compose([
    NormFlattenCast(),
])


class DataServer:

    def __init__(self, hparams: dict) -> None:

        self.dataset = hparams["dataset"]
        self.batch_size =  hparams["batch_size"]
        self.num_targets = hparams["num_targets"]
        self.num_workers = hparams["num_workers"]

        home_dir = Path.home()
        root_dir = f"{home_dir}/data/{self.dataset}"
        # root_dir = "/tmp/" + f"{dataset}"

        if self.dataset == "cifar10":
            self.train_dataset = CIFAR10(root=root_dir, train=True, download=True, transform=train_transforms)
            self.test_dataset = CIFAR10(root=root_dir, train=False, download=True, transform=test_transforms)
        elif self.dataset == "fashion_mnist":
            self.train_dataset = FashionMNIST(root=root_dir, train=True, download=True, transform=train_transforms)
            self.test_dataset = FashionMNIST(root=root_dir, train=False, download=True, transform=test_transforms)
        elif self.dataset == "mnist":
            self.train_dataset = MNIST(root=root_dir, train=True, download=True, transform=train_transforms)
            self.test_dataset = MNIST(root=root_dir, train=False, download=True, transform=test_transforms)
        else:
            raise NotImplementedError(f"Dataset {self.dataset} not available.")

        self.generator = set_generator_seed()

    def get_generator(self):
        """Returns data generator."""
        training_generator = NumpyLoader(
            dataset=self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            worker_init_fn=set_worker_seed,
            generator=self.generator,
        )
        return training_generator

    def get_dataset(self):
        """Returns full training and test dataset."""
        if self.dataset == "cifar10":
            # Get the full train dataset to compute accuray
            train_images = np.array(self.train_dataset.data).reshape(len(self.train_dataset.data), -1)
            train_images = 2.0 * (train_images / 255.0) - 1.0
            train_labels = one_hot(np.array(self.train_dataset.targets), self.num_targets)

            # Get the full test dataset to compute accuray
            test_images = np.array(self.test_dataset.data).reshape(len(self.test_dataset.data), -1)
            test_images = 2.0 * (test_images / 255.0) - 1.0
            test_labels = one_hot(np.array(self.test_dataset.targets), self.num_targets)
        else:
            # Get the full train dataset to compute accuray
            train_images = np.array(self.train_dataset.train_data).reshape(len(self.train_dataset.train_data), -1)
            train_images = 2.0 * (train_images / 255.0) - 1.0
            train_labels = one_hot(np.array(self.train_dataset.train_labels), self.num_targets)

            # Get the full test dataset to compute accuray
            test_images = np.array(self.test_dataset.test_data).reshape(len(self.test_dataset.test_data), -1)
            test_images = 2.0 * (test_images / 255.0) - 1.0
            test_labels = one_hot(np.array(self.test_dataset.test_labels), self.num_targets)

        return train_images, train_labels, test_images, test_labels
