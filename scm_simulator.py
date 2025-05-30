# S_m simulator

# %%

POs = S_m[0]['POs']
# POs = [0,1,2,10,14]
# POs = [0,1,2]
POs = [0, 1, 2, 11]
POs

# %%

from numpy.random import default_rng

rng = default_rng()

print(bound_nodes)
k = 3
print("Generate structural equations for each variable")

f_is = {}
for x_i in bound_nodes:
    nrow = k * 2 * (2 ** len(bound_g.pred[x_i]))  # dom(K) * dom(U_i) *dom(X_pa)
    ncol = len(bound_g.pred[x_i]) + 1 + 1 + 1
    fun_table = np.zeros(shape=(nrow, ncol))
    fun_table = pd.DataFrame(fun_table)
    col_names = []
    col_names.append('X_' + str(x_i))
    col_names.append('Z')
    for p in bound_g.pred[x_i]:
        col_names.append('X_' + str(p))

    col_names.append('U_' + str(x_i))
    fun_table.columns = col_names

    counter = 0
    for z in range(k):
        for pas in product(range(2), repeat=len(bound_g.pred[x_i]) + 1):
            random_f_i_state = (rng.standard_normal(1) > 0)[0]
            values = np.array([random_f_i_state] + [z] + list(pas))
            fun_table.iloc[counter] = values
            counter = counter + 1

    print("f_{}(pa=({}),Z,U_{})".format(x_i, list(bound_g.pred[x_i].keys()), x_i))
    print(fun_table)
    f_is[x_i] = fun_table

print("Generate parameters for exogenenous variables ")

theta_k = []
for xyz in range(k):
    theta_k.append(rng.random())
theta_k = np.array(theta_k) / sum(theta_k)
print('\theta_k')
print(theta_k)

theta_u = []
for u in bound_nodes:
    theta_u.append(rng.random())
print('\theta_u')
print(theta_u)

# %%

print('Permutate exogenous variable states')

nrow = k * 2 ** len(bound_nodes)  # dom(K) * dom(U_i)
ncol = 1 + len(bound_nodes) + 1
event_table = np.zeros(shape=(nrow, ncol))
event_table = pd.DataFrame(event_table)
col_names = []
col_names.append('Z')
for x_i in bound_nodes:
    col_names.append('U_' + str(x_i))
col_names.append('prob')
event_table.columns = col_names

counter = 0
for z in range(k):
    for us in product(range(2), repeat=len(bound_nodes)):

        prob = theta_k[z]
        for i, u_prob in enumerate(theta_u):
            if us[i] == 1:
                prob = prob * theta_u[i]  # U_i == 1
            else:
                prob = prob * (1 - theta_u[i])  # U_i == 0, so 1 - P(u_1)
        values = np.array([z] + list(us) + [prob])
        # print(values)
        # print(event_table)
        event_table.iloc[counter] = values
        counter = counter + 1

print(event_table)

# Propagate state of exogeneous variables to each relevant PO via structural equations with enforced interventions

for PO_ID in POs:

    z_index = 'Z'

    po_index = str("PO=({})".format(PO_ID))

    for x_i in bound_nodes:
        x_i_index = 'X_{}'.format(x_i)
        u_i_index = 'U_{}'.format(x_i)
        po_x_i_index = po_index + "X_{}".format(x_i)

        # x_i = node_id_for_PO_ID(PO_ID)
        # if not x_i == 0:
        #  continue

        event_table[po_x_i_index] = -1

        print('Adding {}'.format(po_x_i_index))

        for i in range(len(event_table)):
            event = event_table.iloc[i, :]
            print('EventID:{}'.format(i))

            z_state = event[z_index]
            u_i_state = event[u_i_index]

            f_i = f_is[x_i].copy()
            f_i = f_i.loc[(f_i[z_index] == z_state) & (f_i[u_i_index] == u_i_state)]

            regime = regime_for_PO_ID(PO_ID)

            # print('regime:{}'.format(regime))
            for pa_i in bound_g.pred[x_i]:
                pa_index = 'X_{}'.format(pa_i)
                pa_x_i_index = "PO=({})".format(PO_ID) + pa_index
                # print(pa_x_i_index)
                # print(event)
                pa_state = event[pa_x_i_index]  # idle or from intervention
                if regime[pa_i] > 0:
                    pa_state = regime[pa_i] - 1
                f_i = f_i.loc[(f_i[pa_index] == pa_state)]
                # print('Filter f_i with (pa_x_i_index, pa_state)=({},{})'.format(pa_x_i_index, pa_state))
                # print(f_i)

            # print(f_i.shape)
            # print(f_i)
            assert len(f_i) == 1
            x_i_state = int(f_i[x_i_index])
            # print("x_i_state={}".format(x_i_state))
            event_table.loc[i, po_x_i_index] = x_i_state

        print('Added {}'.format(po_index))

event_table

# %%

print('Calculate JPT')

nrow = 2 ** len(POs)
ncol = len(POs) + 1
jpt = np.zeros(shape=(nrow, ncol))
jpt = pd.DataFrame(jpt)
jpt.columns = list(POs) + ['prob']

print(jpt)

for i, event in enumerate(product(range(2), repeat=len(POs))):
    for j in range(len(POs)):
        jpt.iloc[i, j] = event[j]
    events = event_table.copy()
    event = jpt.iloc[i, :]

    for PO_ID in POs:
        node_id = node_id_for_PO_ID(PO_ID)
        pa_index = 'X_{}'.format(node_id)
        pa_x_i_index = "PO=({})".format(PO_ID) + pa_index
        events = events[events[pa_x_i_index] == event[PO_ID]]

    # print(events)
    prob = events.loc[:, 'prob'].sum()
    jpt.loc[i, 'prob'] = prob

jpt

# %%

prob_sum = jpt.loc[:, 'prob'].sum()
print(prob_sum)
assert (prob_sum - 1) <= 0.00001

# %%

jpt[jpt.loc[:, 'prob'] > 0]

# %%

jpt

# %%


