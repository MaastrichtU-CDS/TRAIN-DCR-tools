import time
import math
import numpy
import pandas as pd
from sksurv.metrics import concordance_index_censored

from vantage6.tools.util import info


def master(client, data, feature_type, oropharynx, roitype, organization_ids=None,
           coefficients=None, time_col=None, outcome_col=None):
    """Combine partials to global model

    First we collect the parties that participate in the collaboration.
    Then we send a task to all the parties to compute their partial (the
    row count and the column sum). Then we wait for the results to be
    ready. Finally, when the results are ready, we combine them to a
    global average.

    Note that the master method also receives the (local) data of the
    node. In most use cases this data argument is not used.

    The client, provided in the first argument, gives an interface to
    the central server. This is needed to create tasks (for the partial
    results) and collect their results later on. Note that this client
    is a different client than the client you use as a user.
    """

    # Info messages can help you when an algorithm crashes. These info
    # messages are stored in a log file which is sent to the server when
    # either a task finished or crashes.
    info('Collecting participating organizations')

    # Collect all organization that participate in this collaboration.
    # These organizations will receive the task to compute the partial.
    # check if the organizations ids are received as a list
    if isinstance(organization_ids, list) is False:
        organizations = client.get_organizations_in_my_collaboration()
        ids = [organization.get("id") for organization in organizations]
    else:
        ids = organization_ids
    info(f'sending task to organizations {ids}')

    # check if the coefficients ids are received as a list
    if isinstance(coefficients, (list, tuple, numpy.ndarray)) is False:
        info(f'{coefficients} is not of correct datatype, should be list, tuple, numpy array')

    # check if the time column ids are received as a string
    if isinstance(time_col, str) is False:
        info(f'{time_col} is not of correct datatype, should be string')

    # check if the censor column ids are received as a string
    if isinstance(outcome_col, str) is False:
        info(f'{outcome_col} is not of correct datatype, should be string')

    # Request all participating parties to compute their partial. This
    # will create a new task at the central server for them to pick up.
    # We've used a kwarg but is also possible to use `args`. Although
    # we prefer kwargs as it is clearer.
    info('Requesting partial computation')
    task = client.create_new_task(
        input_={
            'method': 'validate_partial',
            'kwargs': {
                'coefficients': coefficients,
                'time_col': time_col,
                'outcome_col': outcome_col,
                'feature_type': feature_type,
                'oropharynx': oropharynx,
                'roitype': roitype
            }
        },
        organization_ids=ids
    )

    # Now we need to wait until all organizations(/nodes) finished
    # their partial. We do this by polling the server for results. It is
    # also possible to subscribe to a websocket channel to get status
    # updates.
    info("Waiting for results")
    task_id = task.get("id")
    task = client.get_task(task_id)
    while not task.get("complete"):
        task = client.get_task(task_id)
        info("Waiting for results")
        time.sleep(1)

    # Once we now the partials are complete, we can collect them.
    info("Obtaining results")
    results = client.get_results(task_id=task.get("id"))

    # this algo only calculates c-index for one node, so only getting the first result
    cindex = 0
    for output in results:
        cindex = output

    return cindex


# gets the dataset from the mentioned path in the node based on the value passed for 'data' arg
def data_selector(data, feature_type, oropharynx, roitype):
    if feature_type == "Clinical":
        df = pd.read_csv('/mnt/data/clinical_data.csv')
        if oropharynx == "yes":
            df = df.loc[df['tumourlocation'] == 'Oropharynx']
        else:
            df = df.loc[df['tumourlocation'] != 'Oropharynx']
        return df
    elif feature_type == "Radiomics":
        df = pd.read_csv('/mnt/data/radiomics_data.csv')
        if oropharynx == "yes":
            df = df.loc[df['tumourlocation'] == 'Oropharynx']
        else:
            df = df.loc[df['tumourlocation'] != 'Oropharynx']
        if roitype == "Primary":
            df = df.loc[df['ROI'] == 'Primary']
        elif roitype == "Node":
            df = df.loc[df['ROI'] != 'Primary']
        return df
    elif feature_type == "Combined":
        df = pd.read_csv('/mnt/data/combined_data.csv')
        if oropharynx == "yes":
            df = df.loc[df['tumourlocation'] == 'Oropharynx']
        else:
            df = df.loc[df['tumourlocation'] != 'Oropharynx']
        return df
    elif feature_type == "LP":
        df = pd.read_csv('/mnt/data/df_lps.csv')
        return df
    else:
        print("Choose the right filters")


def RPC_validate_partial(data, feature_type, oropharynx, roitype, coefficients, time_col, outcome_col):
    """Compute the average partial

    The data argument contains a pandas-dataframe containing the local
    data from the node.
    """

    info(f'Extracting concordance index')
    # ditching the data argument and fetch the normalized dataset from the node
    df = data_selector(data, feature_type, oropharynx, roitype)
    df["lp"] = ""
    lp_list = []
    val_dict = {}
    lp_val = {}

    # extract the beta values from args.
    betas = coefficients[0]
    # print(betas)

    # calculate linear predictor
    lp = 0
    for i, j in df.iterrows():
        for key in betas:
            val_dict[key] = j[key]
            lp_val[key] = (val_dict[key] * betas[key])
        lp = sum(lp_val.values())
        exp_lp = math.exp(lp)
        lp_list.append(exp_lp)

    df['lp'] = lp_list
    # calculate concordance index
    df[outcome_col] = df[outcome_col].astype('bool')
    result = concordance_index_censored(df[outcome_col], df[time_col], df["lp"])
    #cindex = result[0]

    # return the values as a dict
    return {"cindex": result}
