import jax
import jax.numpy as jnp
from jax import random

from jax.experimental.ode import odeint

import optax

import chex

from tqdm import tqdm

import pickle

from neural_function_encoder import init_params
from neural_function_encoder import compute_coefficients_vmap
from neural_function_encoder import evaluate_model


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


def loss_function(params, x, t, y, c):
    """Loss function."""
    xs = y[:, :-1, :]
    ys = y[:, 1:, :]
    t_difs = t[:, 1:] - t[:, :-1]
    # append 0 infront of tdifs, so the last dimension is of size 2, and the first is 0
    t_difs = jnp.concatenate([jnp.zeros((t_difs.shape[0], t_difs.shape[1],  1)), jnp.expand_dims(t_difs, axis=2)], axis=2)

    y_hats = evaluate_model(params, xs, t_difs, c)
    loss = jnp.mean(jnp.linalg.norm(ys - y_hats, ord=2, axis=1) ** 2)

    return loss


rng = random.PRNGKey(1)

n_dim = 2
n_basis = 8
layer_sizes = [n_dim, 100, n_dim]
rng, params = init_params(rng, layer_sizes, n_basis)


n_batches = 500 # num gradient steps
batch_size = 10 # num functions

n_samples = 1000 # samples for coefficients and testing
# n_eval_samples = 10 # unused


# schedule = optax.exponential_decay(
#     init_value=1e-1,
#     transition_steps=n_batches,
#     decay_rate=0.001,
#     transition_begin=1,
# )
optimizer = optax.adam(learning_rate=1e-3)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optimizer,
)
opt_state = optimizer.init(params)

tbar = tqdm(range(n_batches))
for i in tbar:
    rng, key1, key2, key3 = random.split(rng, 4)
    x_data = random.uniform(key1, (batch_size, n_dim), minval=-2.0, maxval=2.0)
    t_diff = random.uniform(key2, (batch_size, n_samples), minval=1e-5, maxval=1e-1)
    # mu = random.uniform(key3, (batch_size,), minval=0.1, maxval=2.0)
    mu = random.uniform(key3, (batch_size,), minval=1.0, maxval=1.0)

    t_data = jnp.cumsum(t_diff, axis=1)
    t_data = jnp.concatenate([jnp.zeros((batch_size, 1)), t_data], axis=1)

    y_data = van_der_pol_model_vmap(x_data, t_data, mu)

    c = compute_coefficients_vmap(x_data, t_diff, y_data, params)

    loss, grad = jax.value_and_grad(loss_function)(params, x_data, t_data, y_data, c)

    # # Generate evaluation data to compute the loss.
    # rng, key1, key2 = random.split(rng, 3)
    # x_eval = random.uniform(key1, (batch_size, n_dim), minval=-5.0, maxval=5.0)
    # t_diff = random.uniform(
    #     key2, (batch_size, n_eval_samples), minval=1e-5, maxval=1e-1
    # )

    # t_eval = jnp.cumsum(t_diff, axis=1)
    # t_eval = jnp.concatenate([jnp.zeros((batch_size, 1)), t_eval], axis=1)

    # y_eval = van_der_pol_model_vmap(x_eval, t_eval, mu)

    # loss, grad = jax.value_and_grad(loss_function)(params, x_eval, t_eval, y_eval, c)

    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)

    tbar.set_description(f"Loss: {loss:2f}")

# Save the model params.
with open("van_der_pol_model_params.pkl", "wb") as f:
    pickle.dump(params, f)
