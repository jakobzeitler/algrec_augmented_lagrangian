import time

import jax.numpy as np
from jax import grad
from jax.example_libraries import optimizers
from jax import jit

import matplotlib.pyplot as plt

class MOM():
    def __init__(self, objective=0, constraints=0, num_constraints=0):
        # optional, setup dict for logging values during optimization
        # -------------------------------------------------------------------------
        self.A_mat = 0
        self.results = {
            "lagrangian": [],
            "grad_norms": [],
            "mu": [],
            "lambda": [],
            "constraint_term": [],
            "objective": [],
            "state": [],  # with state I typically mean the optimization parameters
            "value": [],
        }

        # The temperature parameter that is increases throughout
        self.mu_init = 0.1  # Start with a small value here (increases exponentially)
        self.mu_factor = 1.08  # will need to tune this depending on num_rounds
        self.mu_max = 10  # maybe useful to limit mu to some maximum value, this used to be 100
        # all the other flag values are self explanatory
        self.num_constraints = num_constraints

        self.objective = objective
        self.constraints = constraints

    def plot_results(self, results):
        rounds = range(len(results["lambda"]))
        plt.plot(rounds, results["value"], label='Augmented Lagrangian')
        plt.plot(rounds, results["objective"], label='Objective')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        import matplotlib.ticker as mticker
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        # plt.plot(xrange_inputs, targets, label='target')
        plt.legend()
        plt.show()

    def get_objective(self, x):
        return self.objective(x)

    def get_constraints(self, x):
        constraint_value = 0
        for i in range(self.num_constraints):
            constraint_value += self.constraints[i](x)
        return constraint_value

    def get_psi(self, x, lambd, mu, i):
        constraint = self.constraints[i]
        t = constraint(x)
        sigma = lambd[i]
        # Nocedal Jorge, page 516, Formula 17.56
        if t - mu*sigma <= 0:
            return -sigma * t + (1/2*mu) * t**2
        else:
            return - (mu/2) * sigma**2
        

    def get_augmented_lagrangian(self, x, lambd, mu):
        return (self.get_objective(x) + np.sum(np.array([self.get_psi(x, lambd, mu, i) for i in range(self.num_constraints)])))[0]

    def get_new_batch(self):
        #TODO implement
        1

    def update_lambda(self, x, lambd: np.ndarray,
                  mu: float) -> np.ndarray:
        """Update Lagrangian parameters lambda."""
        return np.maximum(lambd - mu * self.get_constraints(x), 0)

    def lagrangian_value_and_grad(self):
        return value_and_grad()

    def optimize_subproblem(self, x, lambd, mu, subproblem_num_rounds=30):
        opt_init, opt_update, get_params = optimizers.sgd(step_size=1e-2)
        #opt_init, opt_update, get_params = optimizers.adam(step_size=1e-2)
        opt_state = opt_init(x)

        #@jit
        def step(i, opt_state, x):
            p = get_params(opt_state)
            g = grad(self.get_augmented_lagrangian)(p, lambd, mu)
            return opt_update(i, g, opt_state)

        for i in range(subproblem_num_rounds):
            #print(f'Subproblem Step: {i + 1} / {num_steps}')
            opt_state = step(i, opt_state, x)
        return get_params(opt_state)

    # Main optimization method
    # -------------------------------------------------------------------------
    def optimize_augmented_lagrangian(self, num_rounds=10, num_variables=0, mu=0.1, options={}):
        subproblem_num_rounds = 30
        warm_start = {}
        if "warm_start" in options:
            warm_start = options["warm_start"]
        if "subproblem_num_rounds" in options:
            subproblem_num_rounds = options["subproblem_num_rounds"]


        # Set Lagrangian multipliers to zero initially
        lambd = np.zeros((self.num_constraints,1))
        x = np.ones((num_variables,1))
        x = x.at[0].set(0)

        if warm_start:
            self.results = warm_start
            lambd = warm_start["lambda"][-1]
            x = warm_start["state"][-1]
            mu = warm_start["mu"][-1]

        t0 = time.time()
        for i in range(num_rounds):
            t1 = time.time()
            print(f'Round: {i+1} / {num_rounds}', end="")

            self.results["lambda"].append(lambd)
            self.results["mu"].append(mu)

            # Optimize subproblem, i.e., optimize Lagrangian at fixed lambda and mu
            # -------------------------------------------------------------------------
            # If we're assuming gradient-based optimization for the subproblem, then:
            num_batches = 1 #TODO adjust
            for j in range(num_batches):
                #batch = self.get_new_batch()  # whatever data you're optimizing over
                #v, grads = self.lagrangian_value_and_grad(state, batch, lambd, mu)
                #gradients = grad(self.get_augmented_lagrangian)(x, lambd, mu)
                x = self.optimize_subproblem(x, lambd, mu, subproblem_num_rounds)
                # Log the lagrangian for each batch (may not be necessary)
                #self.results["lagrangian"].append(v)
                # Could also log gradient norms for debugging
                #self.results["grad_norms"].append([np.linalg.norm(grad) for grad in gradients])

                # Sum to One



            # optional: post inner optimization logging, e.g., constraint value
            # -------------------------------------------------------------------------
            #constraints = self.get_constraints(state, testdata, lambd, mu)  # compute current
            #objective = self.get_objective(state, testdata, lambd, mu)  # compute current
            #self.results["constraint_term"].append(constraints)
            #self.results["objective"].append(objective)
            self.results["state"].append(x)
            self.results["value"].append(self.get_augmented_lagrangian(x, lambd, mu))
            self.results["objective"].append(self.get_objective(x))

            # update lambda and mu
            # -------------------------------------------------------------------------
            lambd = self.update_lambda(x, lambd, mu)
            mu = np.minimum(mu * self.mu_factor, self.mu_max)

            t2 = time.time()
            print(" {}s (total: {}s, ETR: {}s, objective: {}, value: {})".format(int(t2-t1), int(t2-t0), int((t2-t1)*(num_rounds-i) ), self.results["objective"][i], self.results["value"][i]))

            self.plot_results(self.results)

        print(x)

        return self.results

# Now you have everything you'll need in the results dict