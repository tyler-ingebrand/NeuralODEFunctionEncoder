import jax
import jax.numpy as jnp
from jax import random

from jax.experimental.ode import odeint

import chex


def perceptron(params, x):
    """Perceptron model."""

    y = x

    activation = jax.nn.tanh

    for W, b in params[:-1]:
        y = jnp.dot(y, W) + b
        y = activation(y)

    W, b = params[-1]
    y = jnp.dot(y, W) + b

    return y


def neural_ode_dynamics(x, t, params):
    """Function encoder dynamics."""

    g = perceptron(params, x)

    return g


def neural_ode_model(params, x, t):
    """Neural ODE model."""

    y = odeint(neural_ode_dynamics, x, t, params)

    chex.assert_shape(y, (t.shape[0], x.shape[0]))  # (len(t), n_dim)
    return y


# The following two functions are used to vectorize the neural ODE model.
# The first vectorizes the neural ODE model with respect to the state and times.
neural_ode_model_vmap = jax.vmap(neural_ode_model, in_axes=(None, 0, 0))
# The second vectorizes the neural ODE model with respect to the parameters.
neural_ode_model_params_vmap = jax.vmap(neural_ode_model_vmap, in_axes=(0, None, None))


def init_params(rng, layer_sizes, n_basis):
    """Initialize the parameters of the function encoder."""

    params = []

    for i in range(len(layer_sizes) - 1):
        rng, key1, key2 = random.split(rng, 3)

        W = random.uniform(
            key1,
            (n_basis, layer_sizes[i], layer_sizes[i + 1]),
            minval=-0.5,
            maxval=0.5,
        )
        b = random.uniform(
            key2,
            (
                n_basis,
                layer_sizes[i + 1],
            ),
            minval=-0.5,
            maxval=0.5,
        )

        params.append((W, b))

    return rng, params


@jax.jit
def compute_coefficients(x, t_diff, y, params):
    """Compute the coefficients of the function encoder."""

    n_dim = x.shape[0]
    n_basis = params[0][0].shape[0]

    x_train = y[:-1, :]
    x_train = x_train.reshape(-1, n_dim)
    t_train = t_diff
    t_train = t_train.reshape(-1, 1)
    t_train = jnp.concatenate([jnp.zeros((x_train.shape[0], 1)), t_train], axis=1)
    y_train = y[1:, :]
    y_train = y_train.reshape(-1, n_dim)

    gi = neural_ode_model_params_vmap(params, x_train, t_train)
    gi = gi[:, :, -1, :]
    gi = gi - x_train

    dx = y_train - x_train

    c = jnp.einsum("jd,kjd->k", dx, gi) / x_train.shape[0]

    chex.assert_shape(c, (n_basis,))

    return c


compute_coefficients_vmap = jax.vmap(compute_coefficients, in_axes=(0, 0, 0, None))


@jax.jit
def evaluate_model(params, x, t, c):
    """Evaluate the model."""

    _x = jnp.expand_dims(x, axis=1)
    g = neural_ode_model_params_vmap(params, x, t) - _x

    y_hat = jnp.einsum("jk,kjtd->jtd", c, g) + _x

    return y_hat
