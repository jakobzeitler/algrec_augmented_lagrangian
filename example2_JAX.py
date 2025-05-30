from JAX_MOM import MOM

import jax.numpy as np

def objective(params):
    x1, x2 = params
    return x1 + x2

def constraint(params):
    x1, x2 = params
    c1 = x1*x1 + x2*x2 - 2
    return - np.array([c1])



constraints = [constraint]

mom = MOM(objective, constraints, 1)
mom.optimize_augmented_lagrangian(num_rounds=20,num_variables=2,mu=0.1)

# Plot problem surface
import matplotlib.pyplot as plt
# make data
lambd = np.zeros((1,1))
lambd = lambd.at[(0,0)].set(-1)
xx = np.linspace(-2, 2, 64)
yy = np.linspace(-2, 2, 64)
X, Y = np.meshgrid(xx, yy)
Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)
Z = np.zeros((64,64))
for i, x1 in enumerate(xx):
    for j, x2 in enumerate(yy):
        x = np.array([[x1], [x2]])
        val = mom.get_augmented_lagrangian(x, lambd, 1)
        print(f"{i}{j}{val}")
        Z = Z.at[(i,j)].set(val)
levels = np.linspace(np.min(Z), np.min(Z)+8, 10)
# plot
fig, ax = plt.subplots()
ax.contour(X, Y, Z, levels=levels)
x1 = [x[0] for x in mom.results["state"]]
x2 = [x[2] for x in mom.results["state"]]
ax.plot(-1, -1, marker='X', markersize=15, color="red")
ax.plot(x1, x2, color='green',linewidth=3)
ax.scatter(x1, x2, color='blue')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
plt.show()
1+1