import jax
import jax.numpy as jnp
from jax import random

from jax.experimental.ode import odeint

import chex

import pickle

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec

from neural_function_encoder import compute_coefficients_vmap
from neural_function_encoder import neural_ode_model_params_vmap


def van_der_pol(x, t, mu):
    """Van der Pol oscillator."""

    dx = jnp.array([x[1], mu * (1 - x[0] ** 2) * x[1] - x[0]])

    return dx


def van_der_pol_model(x, t, mu):
    """Van der Pol oscillator model."""

    y = odeint(van_der_pol, x, t, mu)

    chex.assert_shape(y, (t.shape[0], x.shape[0]))  # (len(t), n_dim)
    return y


# The following function is used to vectorize the Van der Pol model.
van_der_pol_model_vmap = jax.vmap(van_der_pol_model, in_axes=(0, 0, 0))

# Load the model params.
with open("van_der_pol_model_params.pkl", "rb") as f:
    params = pickle.load(f)


rng = random.PRNGKey(0)

# Plot an evaluation of the model.
n_example_data = 1000
rng, key1, key2 = random.split(rng, 3)
x_data = jnp.array([[-2.0, 2.0]])
t_diff = random.uniform(key1, (1, n_example_data), minval=1e-5, maxval=1e-1)
t_data = jnp.cumsum(t_diff, axis=1)
t_data = jnp.concatenate([jnp.zeros((1, 1)), t_data], axis=1)

# mu = random.uniform(key2, (1,), minval=0.1, maxval=2.0)
mu = random.uniform(key2, (1,), minval=1.0, maxval=1.0)

y_data = van_der_pol_model_vmap(x_data, t_data, mu)

c = compute_coefficients_vmap(x_data, t_diff, y_data, params)


n_horizon = 1000

x_eval = y_data[:, -1, :]
t_eval = jnp.linspace(0.0, 20.0, n_horizon)
t_eval = t_eval.reshape(1, -1)


# try calculating one step at a time, to see if its different
y_preds2 = jnp.zeros((1, n_horizon, 2))
y_preds2 = y_preds2.at[:, 0, :].set(x_eval)
for time in range(n_horizon):
    t_dif = t_eval[:, time+1] - t_eval[:, time]
    y_current = y_preds2[:, time, :]
    # make it go from 0 to t_dif
    t_dif = jnp.concatenate([jnp.zeros((1, 1)), jnp.expand_dims(t_dif, axis=1)], axis=1)
    g = neural_ode_model_params_vmap(params, y_current, t_dif)
    g = g[:, :, -1, :]
    g = g - y_current
    next_y = jnp.einsum("fk,kfd->d", c, g) + y_current
    y_preds2 = y_preds2.at[:, time+1, :].set(next_y)

_x_eval = jnp.expand_dims(x_eval, axis=1)
g = neural_ode_model_params_vmap(params, x_eval, t_eval) - _x_eval
y_pred = jnp.einsum("jk,kjtd->jtd", c, g) + _x_eval

y_true = van_der_pol_model_vmap(x_eval, t_eval, mu)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)

# Plot the trajectory.
ax.plot(y_data[0, :, 0], y_data[0, :, 1], "C0", label="Example data", lw=1)
ax.plot(y_true[0, :, 0], y_true[0, :, 1], "C2", label="Groundtruth", lw=1)

# Plot the prediction.
# ax.plot(y_pred[0, :, 0], y_pred[0, :, 1], "C1", label="Predicted", lw=1)
ax.plot(y_preds2[0, :, 0], y_preds2[0, :, 1], "C3", label="Predicted2", lw=1)

# Plot the unscaled g.
# g = jnp.einsum("jk,kjtd->kjtd", c, g) + _x_eval
# for i in range(8):
#     ax.plot(g[i, 0, :, 0], g[i, 0, :, 1], "C3", lw=1)
plt.legend()
plt.show()
