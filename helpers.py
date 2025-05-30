import numpy as np
import pandas as pd
import networkx as nx
import matplotlib as plt
from itertools import combinations, product
from debugging_helpers import *

def construct_regimes(nodes):
    # 0: observational/natural/idle
    # otherwise: do(x_i=value) where value is in domains index minus 1, e.g. 1 in domains[1-1] is 0, i.e. do(x_i=0)

    regime_states = domain_size
    regimes = np.array(list(product(range(regime_states), repeat=len(nodes)))).astype(int)
    if regimes.shape != ((regime_states) ** len(nodes), len(nodes)):
        print(((regime_states) ** len(nodes), len(nodes)))
        raise AssertionError(regimes.shape)
    return regimes



def single_intervention_regimes(nodes):
  single_regimes = []
  for i, regime in enumerate(construct_regimes(nodes)):
      if regime[len(regime) - 1] != idle_regime_state:
          # Will not consider regimes with interventions on outcome node
          continue

      intervention_counter = 0
      for j, r in enumerate(regime):
          if r != idle_regime_state:
              intervention_counter = intervention_counter + 1
      if intervention_counter < 2:
          single_regimes.append(regime)

  print("Regimes:")
  print(pd.DataFrame(single_regimes))

  return single_regimes


def non_outcome_intervention_regimes(nodes):
  non_outcome_intervention_regimes = []
  for i, regime in enumerate(construct_regimes(nodes)):
      if regime[-1] != 0:
          # Will not consider regimes with interventions on outcome node
          continue
      non_outcome_intervention_regimes.append(regime)
  print("Regimes:")
  print(pd.DataFrame(non_outcome_intervention_regimes))

  return non_outcome_intervention_regimes



def construct_events(nodes):
    # 0: observational/natural/idle
    # otherwise: do(x_i=value) where value is in domains index minus 1, e.g. 1 in domains[1-1] is 0, i.e. do(x_i=0)

     #domain values for each intervention plus natural state
    events = np.array(list(product(range(domain_size), repeat=len(nodes)))).astype(int)
    if events.shape != ((domain_size) ** len(nodes), len(nodes)):
        print(((domain_size) ** len(nodes), len(nodes)))
        raise AssertionError(regimes.shape)
    return events



def construct_graph_for(p):
  bound_g = nx.DiGraph()
  nodes = list(range(p))
  for i, id in enumerate(nodes):
    bound_g.add_node(id, name = "X" + str(id))
    for parent in nodes[:i]:
      bound_g.add_edge(parent, id)
  nx.draw_networkx(bound_g)
  return bound_g

# We need to be able to iterate all possible regime states
# 0: empty set or idle/natural state
# 1: do(X_i=0)
# 2: do(X_i=1)
def all_regime_permutations(p):
  return np.array(list(product(range(3), repeat=p)))


"""
Central to the whole implementation is the ability to 



*   Identify Potential Outcomes (POs) with a pre-defined unique ID
*   Multiple equivalent POs need to be associated with the same ID
*   Query for PO IDs via a regime state and node of interest

Furthermore


*   Assign a unique ID to each set state in each set in $S_m$
*   Query for these unique IDs via the ID of the set and the event state


"""


# To get the row of a PO, simply conver the regime state from its base3 to base10
def base3tobase10(regime): #regime to PO_ID row
  ans = 0
  for c in regime:
      ans = 3 * ans + c
  return ans

def PO_ID_for(node_id, regime, graph):
  if regime[node_id] != 0:
    return "-" # intervened vars have no POs by definition
  regime = regime.copy()

  # Set nodes that are not parents to idle by default
  for i, r in enumerate(regime):
    if i not in graph.pred[node_id]:
      #print('{} not in pa_{}'.format(i, node_id))
      regime[i] = 0

  #print(regime)
  row = base3tobase10(regime)
  return int(row * len(regime) + node_id)

# To get the node id for a PO ID, simply apply modulo and the remainder will be the node id
def node_id_for_PO_ID(PO_ID, p):
  return PO_ID % p

import math
# To get the regime for a PO ID, ...
def regime_for_PO_ID(PO_ID, p):
  node_id = node_id_for_PO_ID(PO_ID, p)
  regime = np.zeros(shape=(1,p))[0]
  row=str(np.base_repr(math.floor(PO_ID/p),base=3))
  #print(PO_ID-node_id)
  #print(row)
  for j in range(len(row)):
    #print(regime)
    regime[-j-1] = int(row[::-1][j])
  return regime


def test_PO_ID_table(p, bound_nodes, bound_graph):
    log_header('PO ID Lookup Table')
    # Test output for PO_ID_for() and node_id_for_PO_ID()
    lookup_table = pd.DataFrame(np.zeros(shape=(0, 4 * p)))
    entries = []
    for i, regime in enumerate(all_regime_permutations(p)):
        entry = {}
        # print("------------")
        text = str(base3tobase10(regime))
        text = text + str(np.frombuffer(regime.tobytes(), dtype=int))

        entry["F_is"] = regime
        for f, r in enumerate(regime):
            # entry["F{}".format(f)] = r
            1

        text = text + ": PO_IDs:"
        for x in range(p):
            PO_ID = PO_ID_for(x, regime, bound_graph)
            text = text + str(PO_ID) + ","
            entry['PO_{}'.format(x)] = PO_ID

        text = text + ": \t\t node_ids:"
        for x in bound_nodes:
            PO_ID = PO_ID_for(x, regime, bound_graph)
            if PO_ID != "-":
                node_id = node_id_for_PO_ID(PO_ID, p)
                text = text + str(node_id) + ","
                entry["id{}".format(x)] = node_id
            else:
                text = text + PO_ID + ","
                entry["id{}".format(x)] = '-'

        text = text + ": \t regimes:"

        for x in bound_nodes:
            PO_ID = PO_ID_for(x, regime, bound_graph)
            if PO_ID != "-":
                reg = str(regime_for_PO_ID(PO_ID, p))
                text = text + reg
                entry['r{}'.format(x)] = reg
            else:
                text = text + PO_ID + ","
                entry['r{}'.format(x)] = "-"

        entries.append(entry)
        #print(text)
    print(pd.DataFrame(entries).to_string())

# All \Theta_m_i are concatenate into a single vector.
# Therefore, to get the ID for an event state in a set, we need to offset its local ID with the length of all previous \Theta_m_i
def theta_m_ID_for(S_m, Sm_ID, event):
  offset = 0
  for id in range(0,Sm_ID):
    offset = offset + 2**len(S_m[id]['POs'])

  theta_m_ID = float(offset + reduce(lambda a,b: 2*a+b, event))
  assert(theta_m_ID.is_integer())
  return int(theta_m_ID)


# Theta_m has dimensions of length equal all Theta_m_i appended to each other
debug_print = False
from functools import reduce


def calculate_theta_m_dimensions(S_m):
    if not S_m: raise AssertionError("First construct S_m, then come back")
    log_header("Constructing \Theta_m dimensions")
    all_IDs = []
    for i, s in enumerate(S_m):
        IDs = []
        for event in list(product(range(2), repeat=len(s['POs']))):
            ID = theta_m_ID_for(S_m, s['id'], event)
            if (debug_print):
                print('{}:{}'.format(ID, event))
            IDs.append(ID)
        print(s)

        print("Num of Parameters: {} ID range:[{}:{}]".format(len(IDs), int(theta_m_ID_for(S_m, s['id'], np.zeros(len(s['POs'])))),
                                                     int(theta_m_ID_for(S_m, s['id'], np.ones(len(s['POs']))))))
        # print(IDs)
        all_IDs = all_IDs + IDs

    last_ID = -1
    for ID in all_IDs:
        if last_ID + 1 != ID:
            raise AssertionError("Problem with Theta_m:(last,current):{}".format((last_ID, ID)))
        last_ID = ID

    return (1, len(all_IDs))


