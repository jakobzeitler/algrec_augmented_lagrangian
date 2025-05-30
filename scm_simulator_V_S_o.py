from numpy.random import default_rng
import numpy as np
import pandas as pd
from itertools import product

from helpers import construct_graph_for
from data_handlers import get_prob_from_python_simulation


def generate_structural_equations(rng, bound_g, p=3, k=3):
    p = p # number of variables
    k = k # domain of confounder Z

    print("Generate structural equations for each variable")
    f_is = {}
    for x_i in range(p):
        nrow = k * 2 * (2 ** len(bound_g.pred[x_i]))  # dom(K) * dom(U_i) * dom(X_pa)
        ncol = len(bound_g.pred[x_i]) + 1 + 1 + 1 # x_i, z, u_i, x_pa
        fun_table = np.zeros(shape=(nrow, ncol))
        fun_table = pd.DataFrame(fun_table)

        col_names = []
        col_names.append('X_' + str(x_i))
        col_names.append('Z')
        col_names.append('U_' + str(x_i))
        for p in bound_g.pred[x_i]:
            col_names.append('X_' + str(p))
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

    return f_is

def generate_exogeneous_variable_JPT(rng, p, k, bound_g):
    print("Generate parameters for exogenenous variables ")

    theta_k = []
    for xyz in range(k):
        theta_k.append(rng.random())
    theta_k = np.array(theta_k) / sum(theta_k)
    print('\theta_k')
    print(theta_k)

    theta_u = []
    for u in range(p):
        theta_u.append(rng.random())
    print('\theta_u')
    print(theta_u)

    print('Permutate exogenous variable states')

    nrow = k * 2 ** p  # dom(K) * dom(U_is)
    ncol = 1 + p + 1 # k, u_is, prob
    event_table = np.zeros(shape=(nrow, ncol))
    event_table = pd.DataFrame(event_table)

    col_names = []
    col_names.append('Z')
    for x_i in range(p):
        col_names.append('U_' + str(x_i))
    col_names.append('prob')
    event_table.columns = col_names

    counter = 0
    for z in range(k):
        for us in product(range(2), repeat=p):
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
    print(event_table['prob'].sum())
    assert(abs(event_table['prob'].sum()-1) < 0.00000001)

    return event_table

def propagate_exog_on_vars_in_regimes(p, k, fun_table, exog_table, bound_g):
    Fs = {}
    for f_i in range(p-1):
        Fs['F_{}'.format(f_i)] = []
    Z = []
    Us = {}
    for u_i in range(p):
        Us['U_{}'.format(u_i)]= []
        probs = []
    Xs = {}
    for x_i in range(p):
        Xs['X_{}'.format(x_i)] = []

    regimes = np.array(list(product(range(3), repeat=p-1))) - 1
    for regime in regimes:
        for index in range(len(exog_table)):
            row = exog_table.iloc[index]
            for f_i in range(p-1):
                Fs['F_{}'.format(f_i)].append(regime[f_i])
            Z.append(row['Z'])
            probs.append(row['prob'])
            for u_i in range(p):
                Us['U_{}'.format(u_i)].append(row['U_{}'.format(u_i)])

            for x_i in range(p):
                if x_i < p -1:
                    if -1 != regime[x_i]:
                        Xs['X_{}'.format(x_i)].append(regime[x_i])
                        continue
                f_i = fun_table[x_i].copy()
                f_i = f_i[f_i['Z'] == row['Z']]
                f_i = f_i[f_i['U_{}'.format(x_i)] == row['U_{}'.format(x_i)]]
                for x_pa in bound_g.pred[x_i]:
                    x_pa_val = Xs['X_{}'.format(x_pa)][-1]
                    f_i = f_i.loc[f_i['X_{}'.format(x_pa)].isin([x_pa_val]) ]

                response = int(f_i['X_{}'.format(x_i)])
                Xs['X_{}'.format(x_i)].append(response)

    data = {'Z':Z}
    data['prob'] = probs
    data.update(Us)
    data.update(Fs)
    data.update(Xs)
    regime_jpt_tables = pd.DataFrame.from_dict(data)

    return regime_jpt_tables

def propagate_exog_on_vars_in_S_m_regimes(p, k, fun_table, exog_table, bound_g):
    Fs = {}
    for f_i in range(p-1):
        Fs['F_{}'.format(f_i)] = []
    Z = []
    Us = {}
    for u_i in range(p):
        Us['U_{}'.format(u_i)]= []
        probs = []
    POs = {}
    for x_i in range(p):
        Xs['X_{}'.format(x_i)] = []

    regimes = np.array(list(product(range(3), repeat=p-1))) - 1
    for regime in regimes:
        for index in range(len(exog_table)):
            row = exog_table.iloc[index]
            for f_i in range(p-1):
                Fs['F_{}'.format(f_i)].append(regime[f_i])
            Z.append(row['Z'])
            probs.append(row['prob'])
            for u_i in range(p):
                Us['U_{}'.format(u_i)].append(row['U_{}'.format(u_i)])

            for x_i in range(p):
                if x_i < p -1:
                    if -1 != regime[x_i]:
                        Xs['X_{}'.format(x_i)].append(regime[x_i])
                        continue
                f_i = fun_table[x_i].copy()
                f_i = f_i[f_i['Z'] == row['Z']]
                f_i = f_i[f_i['U_{}'.format(x_i)] == row['U_{}'.format(x_i)]]
                for x_pa in bound_g.pred[x_i]:
                    x_pa_val = Xs['X_{}'.format(x_pa)][-1]
                    f_i = f_i.loc[f_i['X_{}'.format(x_pa)].isin([x_pa_val]) ]

                response = int(f_i['X_{}'.format(x_i)])
                Xs['X_{}'.format(x_i)].append(response)

    data = {'Z':Z}
    data['prob'] = probs
    data.update(Us)
    data.update(Fs)
    data.update(Xs)
    regime_jpt_tables = pd.DataFrame.from_dict(data)

    return regime_jpt_tables

def verify_jpts(regime_jpt_tables, p):
    regimes = []
    regimes.append(tuple([-1] * (p - 1)))
    [regimes.append(regime) for regime in product(range(2), repeat=p - 1)]
    print('Dim {}'.format(regime_jpt_tables.shape))
    print(regime_jpt_tables)

    for regime in regimes:
        print('Verify JPT for regime = {}'.format(regime))
        regime_table = regime_jpt_tables.copy()
        for i, reg_indicator_value in enumerate(regime):
            regime_table = regime_table[regime_table['F_{}'.format(i)] == reg_indicator_value]
        if abs(regime_table['prob'].sum() - 1) > 0.0000001:
            raise  AssertionError ('bad JPT integrity: sum to one for {}'.format(regime_table['prob'].sum()))

import random

def enforce_parents_limit(rng, bound_g, max_number_parents=3):
    p = len(bound_g.nodes)

    for x_i in bound_g.nodes:
        if x_i == p - 1:
            # No limit on outcome
            break
        x_pa = bound_g.pred[x_i]
        if len(x_pa) <= max_number_parents:
            print('ENOUGH PARENTS')
            continue
        print('TOO MANY PARENTS')
        to_be_removed = rng.choice(len(x_pa), len(x_pa) - max_number_parents, replace=False)
        print('TO BE REMOVED:{}'.format(to_be_removed))
        for i in to_be_removed:
            bound_g.remove_edge(i, x_i)

    return bound_g

def remove_edges_in(bound_g, remove_edges):
    print('WILL REMOVE EDGES {}'.format(remove_edges))
    for (a, b) in remove_edges:
        print('removing {}'.format((a,b)))
        bound_g.remove_edge(a, b)
    return bound_g

def simulate_scm_with_V_S_o_export(p=3, k=3, seed=420, max_number_parents=None, remove_edges=[]):
    rng = default_rng(seed)
    bound_g = construct_graph_for(p)
    if max_number_parents != None:
        bound_g = enforce_parents_limit(rng, bound_g, max_number_parents=1)

    if len(remove_edges) > 0:
        remove_edges_in(bound_g, remove_edges)

    fun_table = generate_structural_equations(rng,bound_g, p,k )
    exog_table = generate_exogeneous_variable_JPT(rng, p, k, bound_g)
    regime_jpt_tables = propagate_exog_on_vars_in_regimes(p, k, fun_table, exog_table, bound_g)
    verify_jpts(regime_jpt_tables, p)


    regimes = []
    idle_regime = [0]*p
    regimes.append(idle_regime)
    for r in range(p-1):
        regime = idle_regime.copy()
        regime[r] = 1
        regimes.append(regime)
    for regime in regimes:
        for event in product(range(2),repeat=p):
            prob = get_prob_from_python_simulation(regime_jpt_tables, event, regime)
            print("P({}|do({})={}".format(event, regime, prob))

    print("TRUE EFFECTS")
    for regime in product(range(2),repeat=p-1):
        regime = list(regime) + [0]
        for event in product(range(2),repeat=p):
            if regime[0:p-1] != list(event)[0:p-1]:
                #continue
                1
            #prob = get_prob_from_python_simulation(regime_jpt_tables, event, regime)
            #print("P({}|do({})={}".format(event, regime, prob))
            #raise AssertionError('Verify that these are correct')
            1

    return regime_jpt_tables

def simulate_scm_with_S_m_export(p=3, k=3, seed=420, max_number_parents=None, remove_edges=[]):
    rng = default_rng(seed)
    bound_g = construct_graph_for(p)
    if max_number_parents != None:
        bound_g = enforce_parents_limit(rng, bound_g, max_number_parents=1)

    if len(remove_edges) > 0:
        remove_edges_in(bound_g, remove_edges)

    fun_table = generate_structural_equations(rng,bound_g, p, k)
    exog_table = generate_exogeneous_variable_JPT(rng, p, k, bound_g)


    #Propagate vars for S_m
    S_m_jpt_tables = propagate_exog_on_vars_in_S_m_regimes()



    regimes = []
    idle_regime = [0]*p
    regimes.append(idle_regime)
    for r in range(p-1):
        regime = idle_regime.copy()
        regime[r] = 1
        regimes.append(regime)
    for regime in regimes:
        for event in product(range(2),repeat=p):
            prob = get_prob_from_python_simulation(regime_jpt_tables, event, regime)
            print("P({}|do({})={}".format(event, regime, prob))

    print("TRUE EFFECTS")
    for regime in product(range(2),repeat=p-1):
        regime = list(regime) + [0]
        for event in product(range(2),repeat=p):
            if regime[0:p-1] != list(event)[0:p-1]:
                #continue
                1
            #prob = get_prob_from_python_simulation(regime_jpt_tables, event, regime)
            #print("P({}|do({})={}".format(event, regime, prob))
            #raise AssertionError('Verify that these are correct')
            1

    return regime_jpt_tables

simulate_scm_with_V_S_o_export()