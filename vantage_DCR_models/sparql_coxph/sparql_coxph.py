import time
import numpy as np
import os
from scipy.stats import norm
import pandas as pd
from vantage6.tools.util import info
import math
from scipy.linalg import solve


def master(client, data, expl_vars, time_col, outcome_col, feature_type, roitype, oropharynx, organization_ids=None):
    """Combine partials to global model
    """
    # Info messages can help you when an algorithm crashes. These info
    # messages are stored in a log file which is sent to the server when
    # either a task finished or crashes.

    fold = "/mnt/data/"
    MAX_COMPLEXITY = 250000

    timestamp = str(round(time.time()))
    os.mkdir(timestamp)
    n_covs = len(expl_vars)
    epochs = 10

    info('Collecting participating organizations')

    # Collect all organisation that participate in this collaboration unless specified
    if isinstance(organization_ids, list) is False:
        organisations = client.get_organizations_in_my_collaboration()
        ids = [organisation.get("id") for organisation in organisations]
    else:
        ids = organization_ids
    info(f'sending task to organizations {ids}')

    m = len(expl_vars)

    # STEP 2: Ask all nodes to return their preliminary info (unique event times with counts and z sums)

    info("Getting unique event times and counts")
    results = subtaskLauncher(client,
                              ['get_unique_event_times_and_counts',
                               {'time_col': time_col, 'outcome_col': outcome_col,
                                'feature_type': feature_type, 'expl_vars': expl_vars,
                                'oropharynx': oropharynx, 'roitype': roitype},
                               ids])

    unique_time_events = []
    for output in results:
        unique_time_events.append(output["times"])

    D_all = pd.concat(unique_time_events)
    D_all = D_all.groupby(time_col, as_index=False).sum()
    unique_time_events = list(D_all[time_col])

    D_all.to_csv(timestamp + "/cv_D_all.csv", index=False)

    ### Ask all nodes to compute the summed Z statistic
    results = subtaskLauncher(client,
                              ['compute_summed_z',
                               {'expl_vars': expl_vars, 'outcome_col': outcome_col, 'feature_type': feature_type,
                                'oropharynx': oropharynx, 'roitype': roitype},
                               ids])

    z_sum = 0
    for output in results:
        z_sum += output["sum"]

    beta = np.zeros(n_covs)
    beta_old = np.zeros(n_covs)
    delta = 0

    for epoch in range(epochs):

        kwargs = {'expl_vars': expl_vars, 'time_col': time_col, 'beta': beta,
                  'unique_time_events': unique_time_events,
                  'feature_type': feature_type, 'oropharynx': oropharynx, 'roitype': roitype}

        results = subtaskLauncher(client, ['perform_iteration', kwargs, ids])

        summed_agg1 = 0
        summed_agg2 = 0
        summed_agg3 = 0

        for output in results:
            summed_agg1 += np.array(output['agg1'])
            summed_agg2 += np.array(output['agg2'])
            summed_agg3 += np.array(output['agg3'])

        primary_derivative, secondary_derivative = compute_derivatives(summed_agg1, summed_agg2, summed_agg3, D_all,
                                                                       z_sum)

        beta_old = beta
        beta = beta_old - solve(secondary_derivative, primary_derivative)

        delta = max(abs(beta - beta_old))

        if math.isnan(delta):
            info("Delta has turned into a NaN???")
            break

        if delta <= 0.000001:
            info("Betas have settled! Finished iterating!")
            break

    pd.DataFrame(beta).to_csv(timestamp + '/beta.csv')
    pd.DataFrame(secondary_derivative).to_csv(timestamp + '/secondary_derivative.csv')
    # Computing the standard errors
    SErrors = []
    fisher = np.linalg.inv(-secondary_derivative)
    for k in range(fisher.shape[0]):
        SErrors.append(np.sqrt(fisher[k, k]))

    # Calculating P and Z values
    zvalues = (np.exp(beta) - 1) / np.array(SErrors)
    pvalues = 2 * norm.cdf(-abs(zvalues))

    # 95%CI = beta +- 1.96 * SE
    results = pd.DataFrame(
        np.array([np.around(beta, 5), np.around(np.exp(beta), 5), np.around(np.array(SErrors), 5)]).T,
        columns=["Coef", "Exp(coef)", "SE"])
    results['Var'] = expl_vars
    results["lower_CI"] = np.around(np.exp(results["Coef"] - 1.96 * results["SE"]), 5)
    results["upper_CI"] = np.around(np.exp(results["Coef"] + 1.96 * results["SE"]), 5)
    results["Z"] = zvalues
    results["p-value"] = pvalues
    results = results.set_index("Var")

    results.to_csv(timestamp + '/results.csv', index=False)

    return results


def data_selector(data, feature_type, oropharynx, roitype):
    if feature_type == "Clinical":
        df = pd.read_csv('/mnt/data/clinical_data.csv')
        if oropharynx == "yes":
            df = df.loc[df['tumourlocation'] == 'Oropharynx']
        else:
            df = df.loc[df['tumourlocation'] != 'Oropharynx']
        return df
    elif feature_type == 'Radiomics':
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



def compute_derivatives(summed_agg1, summed_agg2, summed_agg3, D_all, z_hat):
    tot_p1 = 0
    tot_p2 = 0

    for index, row in D_all.iterrows():
        # primary
        s1 = row['freq'] * (summed_agg2[index] / summed_agg1[index])

        # secondary
        first_part = (summed_agg3[index] / summed_agg1[index])

        # the numerator is the outerproduct of agg2
        numerator = np.outer(summed_agg2[index], summed_agg2[index])
        denominator = summed_agg1[index] * summed_agg1[index]
        second_part = numerator / denominator

        s2 = row['freq'] * (first_part - second_part)

        tot_p1 += s1
        tot_p2 += s2

    primary_derivative = z_hat - tot_p1
    secondary_derivative = -tot_p2
    return primary_derivative, secondary_derivative


def RPC_perform_iteration(data, expl_vars, time_col, beta, unique_time_events, feature_type, oropharynx, roitype):
    df = data_selector(data, feature_type, oropharynx, roitype)
    D = len(unique_time_events)
    n_covs = len(expl_vars)

    agg1 = []
    agg2 = []
    agg3 = []

    for i in range(D):
        R_i = df[df[time_col] >= unique_time_events[i]][expl_vars]
        # Check if R_i is empty
        if not R_i.empty:
            ebz = np.exp(np.dot(np.array(R_i), beta))
            agg1.append(sum(ebz))
            func = lambda x: np.asarray(x) * np.asarray(ebz)
            z_ebz = R_i.apply(func)
            agg2.append(z_ebz.sum())

            summed = np.zeros((n_covs, n_covs))
            for j in range(len(R_i)):
                summed = summed + np.outer(np.array(z_ebz)[j], np.array(R_i)[j].T)
            agg3.append(summed)

        else:
            agg1.append(0)
            agg2.append(pd.Series(np.zeros(len(expl_vars)), index=expl_vars))
            agg3.append(np.zeros((n_covs, n_covs)))

    return {'agg1': agg1,
            'agg2': pd.DataFrame(agg2),
            'agg3': agg3}


def RPC_compute_summed_z(data, expl_vars, outcome_col, feature_type, oropharynx, roitype):
    df = data_selector(data, feature_type, oropharynx, roitype)
    return {'sum': df[df[outcome_col] == 1][expl_vars].sum()}


def RPC_get_unique_event_times_and_counts(data, time_col, outcome_col, feature_type, oropharynx, roitype, expl_vars):
    df = data_selector(data, feature_type, oropharynx, roitype)
    times = df[df[outcome_col] == 1].groupby(time_col, as_index=False).count()
    times = times.sort_values(by=time_col)[[time_col, outcome_col]]
    times['freq'] = times[outcome_col]
    times = times.drop(columns=outcome_col)
    return {'times': times}


def subtaskLauncher(client, taskInfo):
    method, kwargs_dict, ids = taskInfo

    task = client.create_new_task(
        input_={
            'method': method,
            'kwargs': kwargs_dict

        },
        organization_ids=ids
    )

    task_id = task.get("id")
    task = client.get_task(task_id)
    while not task.get("complete"):
        task = client.get_task(task_id)
        # info("Waiting for results")
        time.sleep(1)
    # Once we know the partials are complete, we can collect them.
    results = client.get_results(task_id=task.get("id"))
    return results  # ['data'][0]['result']