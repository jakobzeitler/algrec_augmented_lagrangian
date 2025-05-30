from helpers import *
import pandas as pd

# Testing R execution

from subprocess import Popen, PIPE

def request_data_from_simulation(p, path, with_S_m, seed = 1, snr=5, sparsity=1):
    clear_simulation_directory(path)

    log_header("Request simulation data from R")
    # Define command and arguments
    command = 'Rscript'
    path2script = './R_python_simulation_bridge.R'

    # Variable number of args in a list
    query_params = '{'
    query_params += '"p":' + str(p)
    query_params += ',"path":"' + path + '"'
    query_params += ', "with_S_m":' + str(int(with_S_m))
    query_params += ',"seed":' + str(seed)
    query_params += ',"snr":' + str(snr)
    query_params += ',"sparsity":' + str(sparsity)
    query_params += '}'
    args = [str(query_params)]

    # Build subprocess command
    cmd = [command, path2script] + args
    print(cmd)
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, error = p.communicate()

    if p.returncode == 0:
        print('R OUTPUT:\n {}'.format(output))
    else:
        print('R ERROR:\n {}'.format(p.returncode))
        print('ERROR:\n {}'.format(error))
        raise AssertionError("R simulation failed. Investigate error message for more.")

    list_simulation_directory(path)
    list_simulation_directory(path)

def list_simulation_directory(path):
    import os
    cwd = os.getcwd()
    print(cwd)
    import os
    log_header("### Data in Simulation Directory ###")
    for entry in os.scandir(path):
        print(entry.name)

def clear_simulation_directory(path):
    import os
    cwd = os.getcwd()
    print(cwd)
    import os
    log_header("### Data in Simulation Directory ###")
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason %s' % (file_path, e))

    for entry in os.scandir(path):
        print(entry.name)

def load_S_o_jpt(path):
    S_o_jpt = pd.read_csv(path)#'S_o_jpt.csv')
    return S_o_jpt

def load_true_joint_effects(path):
    true_effects = pd.read_csv(path)
    return true_effects

def get_prob_from_R_simulation(S_o_data, event, regime):
  data = S_o_data.copy()


  #print(data)
  for i, e in enumerate(event):
    var = 'x' + str(i+1)
    data = data[data[var] == e]

  for i, r in enumerate(regime):
    if i < len(regime)-1:
        if r == 0:
            data = data[data['F' + str(i+1)] == -1]
        if r == 1:
            data = data[data['F' + str(i+1)] == event[i]]

  if data.shape[0] == 1:
    #print(data)
    return data.iloc[0]['prob']
  else:
    print('(regime,event):{}'.format((regime,event)))
    print(data)
    print(S_o_data)
    raise AssertionError("Not supported by R simulation")

def test_S_o_integrity():
    for regime in single_intervention_regimes(bound_nodes):
        print("REGIME={}".format(regime))
        sum_to_one = 0
        for event in product(range(2),repeat=len(bound_nodes)):
            sum_to_one = sum_to_one + get_prob_from_R_simulation(event, regime)
        sum_to_one = sum_to_one / (np.sum(regime) + 1)
        print("SUM={}".format(sum_to_one))
        assert(np.abs(sum_to_one - 1) < 0.0001)


def get_prob_from_python_simulation(regime_jpt_tables, event, regime):
  data = regime_jpt_tables.copy()

  #print(data)
  for i, e in enumerate(event):
    var = 'X_' + str(i)
    data = data[data[var] == e]

  for i, r in enumerate(regime):
    if i < len(regime)-1:
        if r == 0:
            data = data[data['F_' + str(i)] == -1]
        if r == 1:
            data = data[data['F_' + str(i)] == event[i]]

  if data.shape[0] != 0:
    return data['prob'].sum()
  else:
    return 0

def get_true_effect_from_python_simulation(regime_jpt_tables, regime):
  data = regime_jpt_tables.copy()

  # Filter for rows with the right regime, intervened vars need to reflect the regime, idle vars can and should be anything
  for i, r in enumerate(regime):
    data = data[data['F_' + str(i)] == r]
    if r != -1:
      data = data[data['X_{}'.format(i)] == r]

  data = data[data['X_{}'.format(len(regime))] == 1]

  # Now we have all rows that correspond to the event and the regime, so we just need to sum all probabilities
  if data.shape[0] != 0:
    return data['prob'].sum()
  else:
    return 0

"""
This data handlers is for debugging the constraints.

We import a JPT of S_m, i.e. the thruth from a simulator. Constraints that disagree necessarily will have to be wrong by construction.
"""


import pandas as pd
def load_S_m_jpts(path, S_m):
    S_m_JPTs = []
    for S_m_i_ID in range(len(S_m)):
        R_jpt = pd.read_csv(path + '{}.csv'.format(S_m_i_ID+1))
        S_m_JPTs.append(R_jpt)

    for data in S_m_JPTs:
        data = data.copy()
        data = data.drop(data.columns[0], axis=1)
        assert(data.loc[:,'prob'].sum() -1 < 0.0001)

    return S_m_JPTs

def get_crossworld_joint_prob_from_R_simulation(event, S_m_i_ID, S_m_JPTs):
  data = S_m_JPTs[S_m_i_ID].copy()
  data = data.drop(data.columns[0], axis=1)

  for i, e in enumerate(event):
    data = data[data.iloc[:,i] == e]

  #print(len(data))
  if data.shape[0] == 1:
    #print(data)
    return data.iloc[0]['prob']
  else:
    raise AssertionError("Not supported by R simulation")