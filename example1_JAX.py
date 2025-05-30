import jax.numpy as np
from jax import random, value_and_grad, jit
from jax import random
import jax.experimental.optimizers as optim

from typing import Tuple, Text
#Params = Tuple[np.ndarray, np.ndarray, np.ndarray]
Params = Tuple[np.ndarray]

def objective(x1, x2):
    return x1 + x2

def constraint(x1, x2):
    return x1*x1 + x2*x2 - 2


mu = 0.0001
learning_rate = 0.005

tau_init = 10
tau_factor = 5
tau_max = 1e5

seed = 0
key = random.PRNGKey(0)

lmbda = [0.0]
lmbda_history =  []

constraints = [constraint]

def update_lambda(xs, c, prev_lmbda, mu):
    x1, x2 = xs
    if c(x1, x2) <= (-prev_lmbda / mu):
        return 0
    else:
        return prev_lmbda + mu * c(x1, x2)

def theta(xs, c, prev_lmbda, mu):
    x1, x2 = xs
    if c(x1, x2) <= (-prev_lmbda/mu):
        return -(prev_lmbda*prev_lmbda/2*mu)
    else:
        return prev_lmbda * c(x1, x2) + (mu * c(x1, x2) * c(x1, x2)/2)

def augmented_lagrangian(params, lmbda, mu):
    x1, x2 = params[0]
    value = 0
    value += objective(x1, x2)

    xs = (x1, x2)
    for j, c in enumerate(constraints):
        value += theta(xs, c, lmbda_history[-1][j], mu)

    return value

decay_steps = 1000
decay_rate = 1.0
staircase = False
SDG_momentum = 0.9

lagrangian_value_and_grad = value_and_grad(augmented_lagrangian, argnums=1)

epochs = 100
inner_loop_steps = 100
count = 0

def init_params(key: np.ndarray) -> Params:
  """Initiliaze the optimization parameters."""
  key, subkey = random.split(key)
  # init diagonal at 0, because it will be exponentiated
  return np.array([[0.01, 0.02]])
  return 0.05 * np.ones((2,1))

params = init_params(key)
step_size = optim.inverse_time_decay(learning_rate, decay_steps, decay_rate, staircase)
init_fun, update_fun, get_params = optim.sgd(step_size)
state = init_fun(params)

for epoch in range(epochs):
    # log params before first step
    lmbda_history.append(lmbda)

    # a) fix lmbda, mu and optimize for theta using augmented lagrangian
    for j in range(inner_loop_steps):
        value, grads = lagrangian_value_and_grad(get_params(state), lmbda, mu)
        state = update_fun(count, grads, state)
        count =+ 1

    # b) fix theta, mu and optimize for lmbda using update function
    for j, l in enumerate(lmbda):
        lmbda[j] = update_lambda((x1, x2), constraints[j], l, mu)

    mu = np.minimum(mu * tau_factor, tau_max)


