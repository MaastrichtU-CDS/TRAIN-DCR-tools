import time
import math
import numpy
import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc

from vantage6.tools.util import info

def master(client, data, feature_type, oropharynx, roitype, split_data, train_data,
           organization_ids=None, coefficients=None, time_col=None, outcome_col=None):
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
                'roitype': roitype,
                'split_data': split_data,
                'train_data': train_data
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


def data_selector(data, feature_type, oropharynx, roitype, split_data, train_data):
    # Handle LP feature type separately
    if feature_type == "LP":
        prefix = 'train' if train_data else 'test'
        return pd.read_csv(f'/mnt/data/df_lp_{prefix}.csv')

    # Determine file path based on feature type and data split
    if split_data:
        prefix = 'train' if train_data else 'test'
        file_path = f'/mnt/data/{prefix}_{feature_type.lower()}_data.csv'
    else:
        file_path = f'/mnt/data/{feature_type.lower()}_data.csv'

    info(f"Reading data from {file_path}")
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Filter by tumour location
    df = df[df['tumourlocation'] == 'Oropharynx'] if oropharynx == "yes" else df[
        df['tumourlocation'] != 'Oropharynx']

    # Filter by ROI type for Radiomics or Combined feature types
    if feature_type in ['Radiomics', 'Combined']:
        if roitype == "Primary":
            df = df[df['ROI'] == 'Primary']
        elif roitype == "Node":
            df = df[df['ROI'] != 'Primary']

    return df


# Function to calculate bootstrap confidence intervals
def bootstrap_ci(data, feature_type, oropharynx, roitype, coefficients, time_col, outcome_col, split_data,
                 train_data, num_bootstrap=1000, ci=0.95):
    """Calculate bootstrap confidence intervals for cindex."""
    info("Calculating bootstrap confidence intervals")

    # Fetch the normalized dataset
    df = data_selector(data, feature_type, oropharynx, roitype, split_data, train_data)
    df["lp"] = ""
    lp_list = []

    # Extract beta coefficients
    betas = coefficients[0]

    # Calculate linear predictor (lp)
    for i, j in df.iterrows():
        val_dict = {key: j[key] for key in betas}
        lp_val = {key: (val_dict[key] * betas[key]) for key in betas}
        lp = sum(lp_val.values())
        exp_lp = math.exp(lp)
        lp_list.append(exp_lp)
    df['lp'] = lp_list

    # Ensure outcome_col is boolean
    df[outcome_col] = df[outcome_col].astype('bool')

    # Perform bootstrap resampling
    cindex_bootstrap = []
    n = len(df)
    valid_bootstraps = 0
    attempts = 0
    max_attempts = num_bootstrap * 2  # Allow more tries in case of skips

    while valid_bootstraps < num_bootstrap and attempts < max_attempts:
        sample_df = df.sample(n=n, replace=True)
        if sample_df[outcome_col].sum() == 0:
            attempts += 1
            continue  # skip all-censored bootstrap
        try:
            result = concordance_index_censored(
                sample_df[outcome_col],
                sample_df[time_col],
                sample_df["lp"]
            )
            cindex_bootstrap.append(result[0])
            valid_bootstraps += 1
        except Exception as e:
            info(f"Bootstrap sample failed: {e}")
        attempts += 1

    if len(cindex_bootstrap) == 0:
        info("Warning: All bootstrap samples were invalid. Returning None CI.")
        return {"lower_ci": None, "upper_ci": None}

    # Calculate confidence intervals
    lower_bound = np.percentile(cindex_bootstrap, ((1 - ci) / 2) * 100)
    upper_bound = np.percentile(cindex_bootstrap, (1 - (1 - ci) / 2) * 100)

    return {"lower_ci": lower_bound, "upper_ci": upper_bound}


def RPC_validate_partial(data, feature_type, oropharynx, roitype, coefficients, time_col, outcome_col, split_data, train_data):
    """Compute the average partial with confidence intervals."""
    info(f'Extracting concordance index')

    # Fetch the normalized dataset
    df = data_selector(data, feature_type, oropharynx, roitype, split_data, train_data)
    df["lp"] = ""
    lp_list = []
    val_dict = {}
    lp_val = {}

    # Extract the beta coefficients
    betas = coefficients[0]

    # Calculate linear predictor
    for i, j in df.iterrows():
        for key in betas:
            val_dict[key] = j[key]
            lp_val[key] = (val_dict[key] * betas[key])
        lp = sum(lp_val.values())
        exp_lp = math.exp(lp)
        lp_list.append(exp_lp)

    df['lp'] = lp_list

    # Calculate concordance index
    df[outcome_col] = df[outcome_col].astype('bool')
    # ❗ Check for all-censored case (i.e., no events)
    if df[outcome_col].sum() == 0:
        info("All samples are censored — skipping validation for this center.")
        return None

    try:
        result = concordance_index_censored(df[outcome_col], df[time_col], df["lp"])
        cindex = result[0]
    except Exception as e:
        info(f"Failed to compute concordance index: {e}")
        return None

    # Calculate bootstrap confidence intervals
    ci_result = bootstrap_ci(data, feature_type, oropharynx, roitype, coefficients, time_col, outcome_col, split_data, train_data)

    # Return the cindex and confidence intervals
    return {"cindex": cindex, "confidence_intervals": ci_result}
