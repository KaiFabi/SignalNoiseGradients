# Uncertainty-informed Stochastic Gradient Descent 

This project explores uncertainty-informed stochastic gradient descent using JAX by taking the batch's gradients' statistics into account to compute adaptive gradients adjusted with respect to their uncertainty.

## Introduction

Standard stochastic gradient descent (SGD) accumulates the (i.e., takes the sum of gradients over all examples in a batch) gradients for every example of a random batch to compute the per-parameter gradients that are then used to perform the gradient descent step. At the core, more advanced optimizers also use these accumulated gradients which are then fed into a more sophisticated machinery to derive the final gradients used for the optimization step.

Accumulating the gradients is an efficient way to compute approximate gradients. Most major machine learning frameworks do this by default. However, valuable information is getting lost by the summation over all per-example gradients.

Per-example gradients can be used to get information about the variability of gradients for each single model parameter. Thus, per-example gradients allow us to compute the variance or uncertainty associated with every model parameter's gradients. We can use this information about the gradient's variability to adjust the final gradient by taking this uncertainty into accout.

As the variance can be interpreted as a uncertainty associated with a parameter's gradients, the information about the gradients' variability can be used to reduces the gradients magnitude.

## Method

Let $\frac{dL}{dw_{ij}}$ be the gradient of weight $i$ with $i = 1, \dots, N$ for batch sample $j$ with $j = 1, \dots, B$.

Plain stochastic gradient descent (SGD) computes the final gradient for weight $w$ at time step $n+1$ by summing over the individual gradients of the entire batch. Dividing the accumulated gradients by batch size $B$ results in the gradient signal:

$$\mu_i = \frac{dL}{dw_{i}} = \text{E}[\frac{dL}{dw_{ij}}] = \frac{1}{B} \sum_{j=1}^{B} \frac{dL}{dw_{ij}}$$

Instead of reducing the gradients over the entire batch, we can also compute the gradients' variance or squared noise that we can use to adjust the gradients according to the uncertainty in the gradients. Thus, in addition to the average gradient above, we compute also the average gradient's variance

$$\sigma_i^2 = \text{Var}[\frac{dL}{dw_{ij}}] = \frac{1}{B} \sum_{j=1}^{B} (\frac{dL}{dw_{ij}} - \mu_i)^2$$

We can use the information about the gradient's noise to make adjustments to it. In the following are three suggestions for noise-informed gradients

> Noise-informed gradients I
> $$\partial_{w_i} L' = \mu_i \min(\frac{1}{\alpha \cdot \sigma_i}, 1)$$

> Noise-informed gradients II
> $$\partial_{w_i} L' = \mu_i \min(\frac{\mu_i^2}{\alpha \cdot \sigma_i^2}, 1)$$

> Noise-informed gradients III
> $$\partial_{w_i} L' = (1 - \min(\alpha \cdot \sigma_i, 1)) \cdot \mu_i$$

where $\alpha$ is a positive scalar hyperparameter. Adjusted Signal-to-Noise Ratio I+II are modified versions of the signal-to-noise ratio. In both cases, aggregated gradients are kept in case of zero variance and reduced by a factor of $\frac{1}{\alpha \cdot \sigma_i}$ and $\frac{\mu_i^2}{\alpha \cdot \sigma_i^2}$, respectively.

In all cases, gradients adjusted for their noise are small if the variance is high. These adjusted gradients can now be used for standard gradient descent

$$w_i^{n+1} = w_i^n - \eta \partial_{w_i^n} L'$$


## Implementation

I used JAX to implement the method for uncertainty-adjusted gradient computation described above. JAX allows for easy access to the gradients of each sample in the batch. In many other frameworks, such as PyTorch or TensorFlow, it is often not trivial to compute per-example gradients. These libraries often directly accumulate the gradients for each example of the batch to save memory resources.

For more information on sample-wise gradients see also [here](https://jax.readthedocs.io/en/latest/jax-101/04-advanced-autodiff.html#per-example-gradients).

Before we can compute the statistics over a batch of samples, we compute the gradients per-sample which JAX allows to do in an easy but efficient way. For this we combine *jit*, *vmap*, and *grad* transformation.

This implementation of uncertainty adjusted gradients used [this](https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html) JAX tutorial as an entry point.


## Experiment

**Preliminary analysis**

*Experiments have been conducted only for very small networks and datasets due to very limited computational resources.*

In the experiments I compared the method presented in this post to plain SGD and SGD with momentum on a simple classification task.

To empirically evaluate the proposed method of uncertainty-adjusted gradient computation, I applied the method to multilayer fully connected neural networks. The networks consisted of three hidden layers with 1024 neurons each.

The networks are initialized identically when compared to different optimization algorithms. The network's parameters were initialized using the method described in *Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification - He, K. et al. (2015)*, using a normal distribution.

The learning rates as well as the momentum parameter are searched over a dense grid in a case of a single hyperparameter or by applying a random search for the optimization of learning rate and momentum parameter. The reported results stem from the best set of hyper-parameters.

The following learning rates have been found to work best for respective optimizer:

| Optimizer | Learning rate |
|---|:---:|
| SGD | tba |
| SGD + sngrad | tba |

The networks were trained for 200 epochs and a batch size of 128, 256, 512, and 1024.


## Results

**Preliminary analysis**

The following table shows the test accuracy acieved for different datasets.

| Optimizer | SGD | SGD + sngrad
|---|:---:|:---:|
| MNIST   | tba | tba |
| Fashion MNIST | tba | tba |
| CIFAR10  | tba | tba |


## Discussion

Although the results are from very simple examples, they look very promising and required no tweaking of parameters. It seems a bit that uncertainty adjusted gradients work just out of the box.

Efficient low-level implementations of *sngrad* in popular autograd engines should be possible as there are online methods to compute mean and variance of the per-parameter per-example gradients.

Of course, the method has to be applied to more advanced optimizers, larger networks, and more challenging datasets to make better statements about its potential.


## Acknowledgements

I used [this](https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html) tutorial to get started in building basic neural networks with JAX while developing *sngrad*. 


## TODOs

- According to Wikipedia, $\mu / \sigma$ is just an approximation and only valid for non-negative variables. Alternatively one can use $\mu^2 / \sigma^2$ as an adaptive multiplier leading to $\frac{\mu^2}{\sigma^2}\mu = \frac{\mu^3}{\sigma^2}$ as final gradients? https://en.wikipedia.org/wiki/Signal-to-noise_ratio
- Plot noise and signal-to-noise ratio of gradients.
- Check if gradient noise gets better for larger batch sizes.


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