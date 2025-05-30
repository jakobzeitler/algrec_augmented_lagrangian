from JAX_MOM import MOM

import jax.numpy as np
import scm_simulator_V_S_o

full_jpt = scm_simulator_V_S_o.simulate_scm_with_V_S_o_export()
from data_handlers import get_true_effect_from_python_simulation

print("True intervention effects:")
for do_x1 in range(3):
    for do_x2 in range(3):
        print(f"P(Y|do(x1={do_x1-1}),do(x2={do_x2-1}))={get_true_effect_from_python_simulation(full_jpt,[do_x1-1,do_x2-1])}")
jpt = full_jpt
f_states = [-1,-1]
for i, f in enumerate(f_states):
       jpt = jpt[jpt[f"F_{i}"] == f]
print("Observational JPT")
print(jpt)

# Define variables of optimisation problems

# P(R) = P(R_1) * P(R_2|R_1) * P(R_3|R_2)
# P(R) = s_1 * s_2 * s_3
# s_1 \in {0,1}
s1 = {"domain":[0,1]), "parents":[], "function":{0:0,1:1}}
# s_2 \in {0,1,2,3}
s2 = {"domain":[0,1,0,1], "parents":[s1], "function":{0:{0:0,1:0},1:{0:0,1:1},2:{0:1,1:0},3:{0:1,1:1}}}
# s_3 \in {0,1,2,3,4,5,6,7,8,9,10,12,13,14,15}
s3 = {"domain":[0,1,0,1], "parents":[s2], "function":{0:{0:0,1:0},1:{0:0,1:1},2:{0:1,1:0},3:{0:1,1:1}}}
s_is = [s1,s2,s3]
num_variables= sum([len(s["domain"]) for s in s_is])

def binaryToDecimal(n):
    return int(n,2)

def s_i_index(id=1,state=[0],conditioning=[]):
    previous_si = s_is[0:id-1]
    previous_si_offset = sum([len(s["domain"]) for s in previous_si])
    return previous_si_offset + binaryToDecimal("".join([str(i) for i in state+conditioning]))
print(s_i_index(1,[0],[]))
print(s_i_index(2,[0],[0]))
print(s_i_index(3,[1],[1]))

# Define indivator functiont o construct mathematical expressions of Response Function Variables
def indicator(id,x_state,i):
    s_i = s_is[id-1]
    value = x_state[id-1]
    s_i_outcome = s_i["function"][x_state[id-1]]
    if len(s_i["parents"]) > 0:
        s_i_outcome = s_i_outcome[x_state[id-2]]
    return value == s_i_outcome
# Standardised MOM setup procedure

# OBJECTIVE
def objective(params):
    # P(X3=1|do(X1=1),X1=0,X2=0,X=3)
    x_state = [1,0,0]
    objective = 0
    for i, p in enumerate(params):
        if not indicator(1, x_state, i):
            continue
        if not indicator(2, x_state, i):
            continue
        if not indicator(3, x_state, i):
            continue
        print(f"PO-{i}")
        objective += p
    return objective


# CONSTRAINTS
constraints = []

def non_negativity_constraint():
    for p in range(num_variables):
        def constraint(x):
            return (x[p] - 0.00000001)
        constraints.append(constraint)
non_negativity_constraint()

def s_1_sum_constraint(params):
    c = 0
    for val in s1["domain"]:
        c += params[s_i_index(1,[val],[])]
    return (c - 1)
constraints.append(s_1_sum_constraint)
def s_2_sum_constraint():
    for cond in s2["parents"][0]["domain"]:
        def constraint(params):
            c = 0
            for val in s2["domain"]:
                c += params[s_i_index(2, [val], [cond])]
            return (c - 1)
        constraints.append(constraint)
s_2_sum_constraint()
def s_3_sum_constraint():
    for cond in s3["parents"][0]["domain"]:

        def constraint(params):
            c = 0
            for val in s3["domain"]:
                c += params[s_i_index(3, [val], [cond])]
            return (c - 1)
        constraints.append(constraint)#
s_3_sum_constraint()

# Data Matching Constraint
x_states = []
for x1 in range(2):
    for x2 in range(2):
        for x3 in range(2):
            x_states.append([x1,x2,x3])

for x_state in x_states:
    jpt_state = jpt
    for i, x in enumerate(x_state):
       jpt_state = jpt_state[jpt_state[f"X_{i}"] == x]
    #print(f"{x_state}")
    prob = jpt_state["prob"].sum()
    print(f"matching P({x_state})={prob} with POs")
    def constraint(params):
        c = 0
        for i, p in enumerate(params):
            if not indicator(1,x_state,i):
                continue
            if not indicator(2, x_state,i):
                continue
            if not indicator(3, x_state,i):
                continue
            print(f"PO-{i}")
            c+=p
        return (p - prob) # match data


# Initialise and run MOM (Augmented Lagrangian)
mom = MOM(objective, constraints, len(constraints))
mom.optimize_augmented_lagrangian(num_rounds=15,num_variables=num_variables,mu=0.1)

# Plot problem surface
import matplotlib.pyplot as plt
# make data
lambd = np.zeros((1,1))
lambd = lambd.at[(0,0)].set(-1)
xx = np.linspace(-2, 2, 64)
yy = np.linspace(-8, 2, 64)
X, Y = np.meshgrid(xx, yy)
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
#ax.plot(-1, -1, marker='X', markersize=15, color="red")
ax.plot(x1, x2, color='green',linewidth=3)
ax.scatter(x1, x2, color='blue')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
plt.show()

plt.plot(range(len(mom.results["state"])),np.array(mom.results["state"]).T[0].T, label=list(range(num_variables)))
plt.legend()
plt.show()