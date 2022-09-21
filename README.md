# Uncertainty-informed Stochastic Gradient Descent 

This project explores uncertainty-informed stochastic gradient descent using JAX by taking the batch's gradients' statistics into account to compute adaptive gradients adjusted with respect to their uncertainty.

## Introduction

Standard stochastic gradient descent (SGD) accumulates the (i.e., takes the sum of gradients over all examples in a batch) gradients for every example of a random batch to compute the per-parameter gradients that are then used to perform the gradient descent step. At the core, more advanced optimizers also use these accumulated gradients which are then fed into a more sophisticated machinery to derive the final gradients used for the optimization step.

Accumulating the gradients is an efficient way to compute approximate gradients. Most major machine learning frameworks do this by default. However, valuable information is getting lost by the summation over all per-example gradients.

Per-example gradients can be used to get information about the variability of gradients for each single model parameter. Thus, per-example gradients allow us to compute the variance or uncertainty associated with every model parameter's gradients. We can use this information about the gradient's variability to adjust the final gradient by taking this uncertainty into accout.

As the variance can be interpreted as a uncertainty associated with a parameter's gradients, the information about the gradients' variability can be used to reduces the gradients magnitude.

## Method

Let $\frac{dL}{dw_i}$ (or $\partial_{w_i} L$) be the gradient of network parameter $w$ for a batch sample $i$ with $i = 1, \dots, N$.

Given the gradients for parameter $w$ for each sample from a batch, summing over the individual gradients and dividing the accumulated gradients by batch size $N$ results in the gradient's signal:

$$\mu = \frac{1}{N} \sum_{i=1}^{N} \frac{dL}{dw_i}$$

Instead of just reducing the gradients over the entire batch, we can also compute the gradients' variance or squared noise that we can use to adjust the gradients according to their uncertainty. Thus, in addition to the average gradient above, we compute the gradient's variance as usual:

$$\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (\frac{dL}{dw_i} - \mu)^2$$

In the following, three options are shown how information about the variance can be used for uncertainty-informed gradient adjustments.

Noise-informed gradients I
$$\partial_{w} L = \mu \min(\frac{1}{\alpha \cdot \sigma}, 1)$$

Noise-informed gradients II
$$\partial_{w} L = \mu \min(\frac{\mu^2}{\alpha \cdot \sigma^2}, 1)$$

Noise-informed gradients III
$$\partial_{w} L = \mu \cdot (1 - \min(\alpha \cdot \sigma, 1))$$

Here, $\alpha$ is a positive scalar hyperparameter. Options I and II are modified versions of the signal-to-noise ratio. In all options, aggregated gradients are kept in case of zero variance or reduced by the indicated factor. These adjusted gradients can now be used for standard gradient descent

$$w^{n+1} = w^n - \eta \partial_{w^n} L$$


## Implementation

I used JAX to implement the uncertainty-adjusted gradient computation. JAX allows for easy access to the gradients of each sample in a batch. In other frameworks, such as PyTorch or TensorFlow, it is often not trivial to compute per-example gradients. These libraries often directly accumulate the gradients for each example of the batch to save memory resources.

For more information on sample-wise gradients see also [here](https://jax.readthedocs.io/en/latest/jax-101/04-advanced-autodiff.html#per-example-gradients).

Before we can compute the statistics over a batch of samples, we compute the gradients per-sample which JAX allows to do in an easy but efficient way. For this we combine *jit*, *vmap*, and *grad* transformation as shown in the snippet below:

```python
self.backward = jax.jit(jax.vmap(jax.grad(_loss, argnums=0), in_axes=(None, 0, 0)))
```


## Experiment 

The following experiments compared gradient descent with uncertainty-informed gradient adjustments to vanilla SGD on a simple classification task.

To empirically evaluate the proposed method of uncertainty-adjusted gradients, I applied the method to multilayer fully connected neural networks. The networks consisted of ??? hidden layers with ??? neurons each.

Networks are initialized identically when compared to different optimization algorithms. The network's parameters were initialized using the method described in *Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification - He, K. et al. (2015)*, using a normal distribution.

The learning rates for vanilla SGD are searched over a dense grid. The reported results stem from the learning rate associated with the lowest training loss after ??? epoch.

The following learning rates have been found to work best for respective optimizer:

| Optimizer | Learning rate |
|---|:---:|
| SGD | tba |
| SGD + sngrad | tba |

The networks were trained for ??? epochs and a batch size of 128, 256, 512, and 1024.


## Results

The following table shows the test accuracy acieved for different datasets.

| Optimizer | SGD | SGD + sngrad
|---|:---:|:---:|
| CIFAR10  | tba | tba |
| Fashion MNIST | tba | tba |
| MNIST   | tba | tba |


## Discussion

tba

Of course, the method has to be applied to more advanced optimizers, larger networks, and more challenging datasets to make better statements about its potential.


## Acknowledgements

I used [this](https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html) tutorial to get started in building basic neural networks with JAX while developing *sngrad*. 


## TODOs

- Add test


## Citation

If you find this code useful for your research, please cite the following:

```bibtex
@misc{Fischer2022sngrad,
author={Fischer, Kai},
title={signal-to-noise-gradients},
year={2022},
publisher = {GitHub},
journal = {GitHub repository},
howpublished={\url{https://github.com/kaifishr/SignalNoiseGradients}},
}
```


## License

MIT