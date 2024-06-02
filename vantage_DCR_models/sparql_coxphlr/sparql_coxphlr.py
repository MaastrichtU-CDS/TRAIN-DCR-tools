import time
import numpy as np
import os

from os import listdir
from os.path import isfile, join

from itertools import product
import pandas as pd
from vantage6.tools.util import info


def master(client, data, expl_vars, time_col, outcome_col, feature_type, roitype, oropharynx, n_lambda=50, lambda_range=None, beta_start=None,
           epsilon=1 * 10 ^ -8, epochs=200, organization_ids=None):
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

    path = '/mnt/data/'

    ##### load data from previous step folder ######
    data = load_data(path)
    timestamp = str(round(time.time()))
    os.mkdir('/mnt/data/' + timestamp)

    lambda_min_ratio = None


    info('Collecting participating organizations')

   # organizations = client.get_organizations_in_my_collaboration()
   # ids = [organization.get("id") for organization in organizations]

    if isinstance(organization_ids, list) is False:
        organizations = client.get_organizations_in_my_collaboration()
        ids = [organization.get("id") for organization in organizations]
    else:
        ids = organization_ids

    #print(data.keys())
    #if 'sel_expl_vars' in data:
    #    expl_vars = data[f'{roitype}_sel_expl_vars']['sel_expl_vars'].to_list()
    #else:
    #    info('cant find selected vars ')

    n_covs = len(expl_vars)
    info(str(n_covs))

    if beta_start is None:
        beta_start = np.zeros(n_covs)

    if f'{roitype}_train_total_patients' in data:
        ## load means,std, total_patients
        tot_patients = np.array(data[f'{roitype}_train_total_patients'])[0, 1]
    else:
        kwargs_dict = {'expl_vars': expl_vars + [outcome_col], 'feature_type': feature_type,
                       'oropharynx': oropharynx, 'roitype': roitype, 'time_col': time_col}
        method = 'average_partial'
        # results = subtaskLauncher(client, [method, kwargs_dict, ids])

        task = client.create_new_task(
            input_={
                'method': method,
                'kwargs': kwargs_dict
            },
            organization_ids=ids
        )

        info("Waiting for results")
        task_id = task.get("id")
        task = client.get_task(task_id)
        while not task.get("complete"):
            task = client.get_task(task_id)
            info("Waiting for results")
            time.sleep(1)

        info("Obtaining results")
        results = client.get_results(task_id=task.get("id"))

        # Now we can combine the partials to a global average.
        global_sum = 0
        global_count = 0

        for output in results:
            global_sum += output["sum"]
            global_count += output["count"]

        average = global_sum / global_count
        tot_patients = global_count

    ## STEP 2: Ask all nodes to return their preliminary info (unique event times with counts, maximum regolarization lambda)

    info("Getting unique event times and counts")
    results = subtaskLauncher(client,
                              ['get_unique_event_times_and_counts', {'time_col': time_col, 'outcome_col': outcome_col,
                                                                     'feature_type': feature_type, 'expl_vars': expl_vars,
                                                                     'oropharynx': oropharynx, 'roitype': roitype},
                               ids])

    unique_time_events = []
    for output in results:
        unique_time_events.append(output["times"])

    D_all = pd.concat(unique_time_events)
    D_all = D_all.groupby(time_col, as_index=False).sum()
    unique_time_events = list(D_all[time_col])

    D_all.to_csv('/mnt/data/' + timestamp + "/D_all.csv", index=False)

    # maximum regularization parameter allowed
    info("Getting maximum regularization parameter")
    kwargs = {'expl_vars': expl_vars, 'time_col': time_col, 'beta': np.zeros(n_covs),
              'D_all': D_all, 'feature_type': feature_type, 'oropharynx': oropharynx, 'roitype': roitype}
    results = subtaskLauncher(client, ['compute_exp_eta_sum', kwargs, ids])

    R = []
    for output in results:
        R.append(output["R"])

    # MASTER: fa il merge di R_list per calcolare R
    R = {D_all.iloc[i][time_col]: sum([r[unique_time_events[i]] for r in R]) for i in
         range(len(D_all))}

    pd.DataFrame([R]).T.reset_index().to_csv('/mnt/data/' + timestamp + "/R.csv")

    kwargs = {'expl_vars': expl_vars, 'time_col': time_col, 'outcome_col': outcome_col, 'beta': np.zeros(n_covs),
              'D_all': D_all, "R": R, 'oropharynx': oropharynx, 'feature_type': feature_type, 'roitype': roitype}
    results = subtaskLauncher(client, ['compute_wxz', kwargs, ids])

    wxz = []
    for output in results:
        wxz.append(output["wxz"])

    wxz = sum(wxz)
    #print(wxz)
    pd.DataFrame(wxz).to_csv('/mnt/data/' + timestamp + "/wxz.csv")
    lambda_max = np.max(abs(wxz)) / tot_patients

    info(str(lambda_max))
    if lambda_min_ratio is None:
        if tot_patients < len(expl_vars):
            lambda_min_ratio = 0.01
        else:
            lambda_min_ratio = 0.0001

    lambda_min = lambda_min_ratio * lambda_max
    lambda_seq = np.exp(np.linspace(start=np.log(lambda_max), stop=np.log(lambda_min), num=n_lambda))

    if lambda_range is not None:
        lambdaSel = np.arange(lambda_range[0], lambda_range[1])
        lambda_seq = lambda_seq[lambdaSel]

    pd.DataFrame(lambda_seq, columns=["Path"]).to_csv('/mnt/data/' + timestamp + '/regularizationPath.csv')

    model_list = []
    model = {}
    beta = beta_start
    i = 0
    for l1 in lambda_seq:

        if l1 == lambda_max:
            model[l1] = pd.DataFrame(np.array([np.zeros(n_covs), np.ones(n_covs)]).T, columns=['coef', 'exp(coef)'])
            model[l1].to_csv('/mnt/data/' + timestamp + "/coxphl1_coeff_" + str(l1) + ".csv")
            model_list.append(pd.DataFrame(np.array([np.zeros(n_covs)]).T, columns=[l1]))
            i = 1
        else:
            info("Starting iterations ...")
            delta = 0
            for epoch in range(epochs):

                beta_old = beta.copy()
                coord_order = np.random.permutation(n_covs)
                for coord_index in coord_order:
                    coord = expl_vars[coord_index]

                    print(beta.shape)

                    kwargs = {'expl_vars': expl_vars, 'time_col': time_col, 'beta': beta, 'roitype': roitype,
                              'D_all': D_all, 'oropharynx': oropharynx, 'feature_type': feature_type}
                    results = subtaskLauncher(client, ['compute_exp_eta_sum', kwargs, ids])

                    R = []
                    for output in results:
                        R.append(output["R"])

                    # MASTER: fa il merge di R_list per calcolare R
                    R = {D_all.iloc[i][time_col]: sum([r[unique_time_events[i]] for r in R]) for i in
                         range(len(D_all))}

                    kwargs = {'expl_vars': expl_vars, 'time_col': time_col, 'outcome_col': outcome_col, 'beta': beta,
                              'D_all': D_all, 'coord': coord, 'coord_index': coord_index, 'R': R,
                              'oropharynx': oropharynx, 'feature_type': feature_type, 'roitype': roitype}

                    results = subtaskLauncher(client, ['compute_update_parts', kwargs, ids])

                    numerator = 0
                    denominator = 0
                    for output in results:
                        numerator += output["numeratore"]
                        denominator += output["denominatore"]
                    alpha = 1
                    beta[coord_index] = proxL1Norm(numerator / tot_patients, l1 * alpha) / (
                            denominator / tot_patients + l1 * (1 - alpha))

                delta = np.max(np.absolute(beta - beta_old))
                print(beta)
                info(
                    '[' + str(i) + '/' + str(len(lambda_seq)) + '] ' + str(l1) + ': ' + str(epoch) + ' - ' + str(delta))

                if delta <= epsilon:
                    info("Betas have settled! Finished iterating!")
                    break

            model[l1] = pd.DataFrame(np.array([beta, np.exp(beta)]).T, columns=['coef', 'exp(coef)'])
            model[l1].to_csv('/mnt/data/' + timestamp + "/coxphl1_coeff_" + str(l1) + ".csv")
            model_list.append(pd.DataFrame(np.array([beta]).T, columns=[l1]))
            #print(model_list)
            pd.concat(model_list, axis=1).to_csv('/mnt/data/' + timestamp + "/path.csv")

            i += 1

        if len(np.where(beta == 0)[0]) == 0:
            info('All coefficient different from zero! Feature selection terminated')
            break

    model_df = pd.concat(model_list, axis=1)
    model_df['Coef'] = expl_vars

    return {'model': model_df.set_index('Coef')}

def load_data(path):
    try:
        files = [f for f in listdir(path) if isfile(join(path, f))]
    except Exception as e:
        info(e)
        files = []
    data = {}
    for file in files:
        try:
            table_name = file[:-4]
            data[table_name] = pd.read_csv(path + file)
        except Exception as e:
            info(str(e))
            pass
    return data


def data_selector(data, feature_type, oropharynx, roitype, expl_vars):
    if feature_type == 'Clinical':
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
        if roitype == "Primary":
            df = df.loc[df['ROI'] == 'Primary']
        elif roitype == "Node":
            df = df.loc[df['ROI'] != 'Primary']
        return df
    else:
        print("Choose the right filters")


def proxL1Norm(x, kappa):
    return np.maximum(0., x - kappa) - np.maximum(0., -x - kappa)


def RPC_compute_update_parts(data, expl_vars, time_col, outcome_col, D_all, R, beta, coord, coord_index, feature_type, oropharynx, roitype):
    df = data_selector(data, feature_type, oropharynx, roitype, expl_vars)
    #df.drop(df.loc[df[time_col] == 0].index, inplace=True)
    df = df.dropna(axis=0)
    df = calculate_w_z(D_all, R, beta, outcome_col, df, expl_vars, time_col)
    df['numeratore'] = df['w'] * df[coord] * (df['z'] - (df['eta'] - df[coord] * beta[coord_index]))
    df['denominatore'] = df['w'] * df[coord] ** 2
    numeratore, denominatore = df[['numeratore', 'denominatore']].sum()

    return {'numeratore': numeratore, 'denominatore': denominatore}


def RPC_compute_wxz(data, expl_vars, time_col, outcome_col, beta, D_all, R, feature_type, oropharynx, roitype):
    df = data_selector(data, feature_type, oropharynx, roitype, expl_vars)
    #df.drop(df.loc[df[time_col] == 0].index, inplace=True)
    df = df.dropna(axis=0)
    df = calculate_w_z(D_all, R, beta, outcome_col, df, expl_vars, time_col)
    wxz = np.sum(np.array(df[expl_vars]) * np.array(df['w'])[:, None] * np.array(df['z'])[:, None], axis=0)
    return {'wxz': wxz}


def calculate_w_z(D_all, R, beta, outcome_col, df, expl_vars, time_col):
    D_all['key'] = 1
    df['key'] = 1
    df['eta'] = np.dot(df[expl_vars], beta)
    df['exp_eta'] = np.exp(df['eta'])
    cols = list(df)
    df = pd.merge(df, D_all, on='key', suffixes=('', '_y'))
    df = df[df[time_col + '_y'] <= df[time_col]]
    df['R'] = df[time_col + '_y'].map(R)
    df['w'] = df['freq'] * (df['exp_eta'] * df['R'] - df['exp_eta'] ** 2) / df['R'] ** 2
    df['prik'] = df['freq'] * df['exp_eta'] / df['R']
    df = df.groupby(by=cols, as_index=False, dropna=False).sum()
    df['z'] = df['eta'] + (df[outcome_col] - df['prik']) / df['w']
    return df


def RPC_compute_exp_eta_sum(data, expl_vars, time_col, beta, D_all, feature_type, oropharynx, roitype):
    df = data_selector(data, feature_type, oropharynx, roitype, expl_vars)
    #df.drop(df.loc[df[time_col] == 0].index, inplace=True)
    df = df.dropna(axis=0)
    R = {D_all.iloc[i][time_col]: sum(np.exp(np.dot(df[df[time_col] >= D_all.iloc[i][time_col]][expl_vars], beta))) for
         i in range(len(D_all))}
    return {"R": R}


def RPC_get_unique_event_times_and_counts(data, expl_vars, time_col, outcome_col, feature_type, oropharynx, roitype):
    df = data_selector(data, feature_type, oropharynx, roitype, expl_vars)
    #df.drop(df.loc[df[time_col] == 0].index, inplace=True)
    df = df.dropna(axis=0)
    times = df[df[outcome_col] == 1].groupby(time_col, as_index=False).count()
    times = times.sort_values(by=time_col)[[time_col, outcome_col]]
    times['freq'] = times[outcome_col]
    times = times.drop(columns=outcome_col)
    return {'times': times}

def RPC_average_partial(data, expl_vars, feature_type, oropharynx, roitype, time_col):
    """Compute the average partial

    The data argument contains a pandas-dataframe containing the local
    data from the node.
    """
    df = data_selector(data, feature_type, oropharynx, roitype, expl_vars)
    #df.drop(df.loc[df[time_col] == 0].index, inplace=True)
    df = df.dropna(axis=0)
    # compute sum and N.
    # info(f'Extracting  {column_name}')

    local_sum = df[expl_vars].sum()
    local_count = len(df)
    print(local_sum, local_count)
    # return the values as a dict
    return {
        "sum": local_sum,
        "count": local_count
    }


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
