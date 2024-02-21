from functools import partial

import jax
import jax.numpy as jnp
from jax import random
from jax.nn import relu
from jax.experimental import checkify

from jax.experimental.ode import odeint

import optax

from chex import dataclass

from function_encoder import perceptron, perceptron_vmap
from function_encoder import model, model_vmap
from function_encoder import _basis, _basis_vmap
from function_encoder import loss_function, loss_function_vmap
from function_encoder import init_params

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec


def f(x, C):
    return jnp.dot(x ** jnp.arange(3), C)


def generate_random_coefficients(rng):
    """Generate coefficients for a random quadratic function."""

    rng, key = random.split(rng)

    C = random.normal(key, (3, 1))

    return rng, C


def generate_data(rng, C, n_samples=1000):
    """Generate data from a random quadratic function with Gaussian noise."""

    rng, key1, key2 = random.split(rng, 3)

    x = random.uniform(key1, (n_samples, 1), minval=-5.0, maxval=5.0)

    y = f(x, C) + random.normal(key2, (n_samples, 1))

    return rng, x, y


def generate_data_test(rng, C, n_samples=1000):
    """Generate data from a random quadratic function with Gaussian noise."""

    rng, key1, key2 = random.split(rng, 3)

    x = random.uniform(key1, (n_samples, 1), minval=-5.0, maxval=5.0)

    y = f(x, C)  # + random.normal(key2, (n_samples, 1))

    return rng, x, y


@jax.jit
def update(params, opt_state, x, y, data):
    """Update the parameters of the function encoder."""

    loss, grads = jax.value_and_grad(loss_function)(params, x, y, data)

    # Gradient accumulation.

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return loss, params, opt_state


rng = random.PRNGKey(0)

# learning_rate = 1e-3
# optimizer = optax.adam(learning_rate=learning_rate)

schedule = optax.exponential_decay(
    init_value=1e-2, transition_steps=1000, decay_rate=0.1, transition_begin=1
)
optimizer = optax.adamw(learning_rate=schedule)

n_dims = 1
n_basis = 10
layer_sizes = [1, 32, [n_basis, n_dims]]

n_iters = 1000

rng, params = init_params(rng, layer_sizes)
opt_state = optimizer.init(params)

# Initialize figure.
# Plot a figure with two subplots, one above the other. The first one should be square,
# the second one should be narrow.
fig = plt.figure(figsize=(8, 8))

# Create a gridspec.
gs = GridSpec(4, 3, figure=fig)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[1, 2])
ax7 = fig.add_subplot(gs[2, 0])
ax8 = fig.add_subplot(gs[2, 1])
ax9 = fig.add_subplot(gs[2, 2])
ax10 = fig.add_subplot(gs[3, :])

ax = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]

# Initialize plot.
for i in range(9):
    ax[i].set_xlim(-2.0, 2.0)
    ax[i].set_ylim(-10.0, 10.0)

    ax[i].grid()

# ax[0].set_xlabel("x")
# ax[0].set_ylabel("y")

for i in range(9):
    # Remove tick labels.
    ax[i].set_xticklabels([])
    ax[i].set_yticklabels([])

ax[-1].set_xlim(0, n_iters)
ax[-1].set_ylim(1e-4, 1.0)
ax[-1].set_yscale("log")

ax[-1].set_xlabel("Iteration")
ax[-1].set_ylabel("Loss")

ax[-1].grid()

# Generate data.
n_samples = 100

C = []
example_data = []
x = []
y = []

fe_lines = []

x_plt = jnp.linspace(-5.0, 5.0, 1000).reshape(-1, 1)

for i in range(9):
    rng, C_ = generate_random_coefficients(rng)
    C.append(C_)

    rng, x_, y_ = generate_data_test(rng, C_, n_samples=n_samples)
    data_ = (x_, y_)

    example_data.append(data_)
    x.append(x_)
    y.append(y_)

    # basis_lines = []
    # for i in range(n_basis):
    #     (basis_line,) = ax[0].plot([], [], lw=1, color="gray", zorder=1)
    #     basis_lines.append(basis_line)

    # Plot true function.
    y_plt = f(x_plt, C_).squeeze()
    ax[i].plot(x_plt, y_plt, color="C0", lw=2, label="True function", zorder=2)

    # Plot data.
    ax[i].plot(x_, y_, ".", color="gray", label="Data", zorder=10)

    # Plot function encoder.
    (fe_line_,) = ax[i].plot([], [], lw=2, color="C1", zorder=20)

    fe_lines.append(fe_line_)


# Plot loss.
(loss_line,) = ax[-1].plot([], [], lw=2, color="C0")


loss = jnp.zeros(shape=(n_iters,))

model_plt_vmap = jax.vmap(model, in_axes=(None, 0, (None, 1)))


def animate(i):
    global rng, params, opt_state, loss, x_plt, example_data

    # Sample new data.
    rng, C_ = generate_random_coefficients(rng)
    rng, x_, y_ = generate_data(rng, C_, n_samples=n_samples)
    data_ = (x_, y_)
    loss_, params, opt_state = update(params, opt_state, x_, y_, data_)

    loss = loss.at[i].set(loss_)

    # Update line.
    for j in range(9):
        y_plt = model_vmap(params, x_plt, example_data[j]).squeeze()

        fe_lines[j].set_data(x_plt, y_plt)

    # # Update basis functions.
    # x_ = jnp.linspace(-5.0, 5.0, 1000).reshape(-1, 1)
    # y_ = _basis_vmap(params, x_, data).squeeze()

    # for j in range(n_basis):
    #     basis_lines[j].set_data(x_, y_[:, j])

    # Update loss.
    loss_line.set_data(jnp.arange(i + 1), loss[: i + 1])

    return (*fe_lines, loss_line)


anim = animation.FuncAnimation(
    fig, animate, frames=n_iters, interval=10, blit=True, repeat=False
)

# plt.show()

anim.save("example_polynomial.gif", writer="pillow", fps=120)
