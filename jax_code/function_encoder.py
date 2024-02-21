r"""Function encoder example.

A demo implementation of the function encoder.

A function encoder is a linear regression model that has the form:

.. math::

    f(x) = \sum_{i=1}^m c_i g_i(x)
    
where :math:`c_i` are real coefficients and :math:`g_i(x)` are basis functions
represented by a multi-headed neural network.

The coefficients :math:`c_i` are computed via the following equation:

.. math::

    c_i &= \langle f(x), g_i(x) \rangle
        &= \frac{1}{m} \sum_{j=1}^n f(x_j) g_i(x_j)

where :math:`\langle \cdot, \cdot \rangle` is the inner product and :math:`n` is
the number of samples.

So given a dataset :math:`\{(x_1, y_1), \dots, (x_n, y_n)\}`, we can compute the
coefficients :math:`c_i` as follows:

.. math::

    c_i &= \langle f(x), g_i(x) \rangle
        &= \frac{1}{m} \sum_{j=1}^n y_j g_i(x_j)

we then evaluate the function encoder at a new point :math:`x` as follows:

.. math::

    f(x) = \sum_{i=1}^m c_i g_i(x)

In other words, if :math:`c` is a vector of coefficients in :math:`\mathbb{R}^m`
then the output of the neural network is a matrix :math:`G` of shape
:math:`m \times n` where :math:`G_{ij} = g_i(x_j)`. 

"""

import jax
import jax.numpy as jnp

from jax import random
from jax.experimental import checkify


def perceptron(params, x):
    """Perceptron model.

    Models the basis functions :math:`g_i(x)` as a multi-headed neural network.

    Args:
        params: Parameters of the perceptron.
        x: Input.

    Returns:
        The perceptron evaluated at x.

    """

    y = x

    activation = jax.nn.tanh

    for W, b in params[:-1]:
        y = jnp.dot(y, W) + b
        y = activation(y)

    W, b = params[-1]
    # For the last layer, W is a tensor.
    y = jnp.tensordot(y, W, axes=1) + b
    # y = activation(y)

    return y


def model(params, x, data):
    """Function encoder model.

    Args:
        params: Parameters of the function encoder.
        x: Input.
        data: A tuple containing the input and output data.

    Returns:
        The function encoder evaluated at x.

    """

    # X, Y = data
    _x = data[0]  # (n_samples, n_dim)
    _y = data[1]  # (n_samples, n_dim)

    # Compute the output of the neural network at each data point.
    g = perceptron_vmap(params, _x)  # (n_samples, n_basis, n_dim)

    n = _x.shape[0]

    # Compute the basis functions.
    gx = perceptron(params, x)  # (n_basis, n_dim)

    # Compute the coefficients.
    # c is a vector of real coefficients.
    # _y has dims (n_samples, n_dim)
    # g has dims (n_samples, n_basis, n_dim)
    # Multiply _y by g and sum over the number of samples.
    # c = jnp.einsum("ij,ikj->kj", _y, g) / n  # (n_basis, n_dim)

    # Compute c via the regularized inverse.
    G = jnp.sum(g, axis=2)
    W = jnp.dot(G.T, G)
    W = W + jnp.eye(W.shape[0]) * (1 / n)
    c = jnp.linalg.solve(
        W, (_y.T @ G).T
    )  # (n_samples, n_dim) @ (n_samples, n_basis, n_dim)

    # Compute the output of the function encoder.
    # c has dims (n_basis, n_dim)
    # gx has dims (n_basis, n_dim)
    # Sum over the number of basis functions.
    y_hat = jnp.einsum("ij,ij->j", c, gx)  # (n_dim,)

    return y_hat


def _basis(params, x, data):
    """Individual basis model.

    Args:
        params: Parameters of the function encoder.
        x: Input.
        X: Input data.
        Y: Output data.

    Returns:
        The individual basis evaluated at x.

    """

    # X, Y = data
    X = data[0]
    Y = data[1]

    g = perceptron_vmap(params, X)

    n = X.shape[0]

    gx = perceptron(params, x)

    c = jnp.einsum("ij,ikj->kj", Y, g) / n

    y_hat = c * gx

    return y_hat


def init_params(rng, layer_sizes):
    """Initialize the parameters of the function encoder."""

    params = []

    for i in range(len(layer_sizes) - 2):
        rng, key1, key2 = random.split(rng, 3)

        W = random.uniform(
            key1, (layer_sizes[i], layer_sizes[i + 1]), minval=-0.5, maxval=0.5
        )
        b = random.uniform(key2, (layer_sizes[i + 1],), minval=-0.5, maxval=0.5)

        params.append((W, b))

    # The last layer is a tensor.
    rng, key1, key2 = random.split(rng, 3)

    W = random.uniform(
        key1,
        (layer_sizes[-2], layer_sizes[-1][0], layer_sizes[-1][1]),
        minval=-0.5,
        maxval=0.5,
    )
    b = random.uniform(key2, (layer_sizes[-1][1],), minval=-0.5, maxval=0.5)

    params.append((W, b))

    return rng, params


def loss_function(params, x, y, data):
    y_hat = model_vmap(params, x, data)

    n = y.shape[0]
    loss = (1 / n) * jnp.linalg.norm(y - y_hat, ord=2)

    return loss


perceptron_vmap = jax.vmap(perceptron, in_axes=(None, 0))

model_vmap = jax.vmap(model, in_axes=(None, 0, (None, None)))

_basis_vmap = jax.vmap(_basis, in_axes=(None, 0, (None, None)))

loss_function_vmap = jax.vmap(loss_function, in_axes=(None, 0, 0, (1, 1)))
