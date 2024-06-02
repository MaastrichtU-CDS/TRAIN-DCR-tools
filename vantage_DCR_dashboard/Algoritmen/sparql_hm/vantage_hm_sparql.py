import requests
import time
from itertools import product
import numpy as np
import pandas as pd

from vantage6.tools.util import info
from io import StringIO


def master(client, data, roitype, expl_vars, censor_col, organization_ids=None):
    """
    Combine partials to global model for the provided predicate and specified organisations

    :param client: vantage client
    :param data: data defined in node config file
    :param predicate: predicate to retrieve the average from, defined by user
    :param organization_ids: organisations to run the algorithm on, defined by user
    :return: global average (global_sum / global_count)
    """
    # The info messages are stored in a log file which is sent to the server when either a task finished or crashes
    info('Collecting participating organizations')

    # Collect all organisation that participate in this collaboration unless specified
    if isinstance(organization_ids, list) is False:
        organisations = client.get_organizations_in_my_collaboration()
        ids = [organisation.get("id") for organisation in organisations]
    else:
        ids = organization_ids
    info(f'sending task to organizations {ids}')

    # Request all (specified) organisations to compute their partial
    info('Requesting partial average using sparql')
    task = client.create_new_task(
        input_={
            'method': 'average_partial',
            'kwargs': {'expl_vars': expl_vars,
                       'roitype': roitype}
        },
        organization_ids=ids
    )

    # The result is awaited by polling the server for results
    info("Waiting for results")
    task_id = task.get("id")
    task = client.get_task(task_id)
    while not task.get("complete"):
        task = client.get_task(task_id)
        info("Waiting for results")
        time.sleep(1)

    # Collect the results
    info("Obtaining results")
    results = client.get_results(task_id=task.get("id"))

    # Initialize an empty set to store unique features
    global_features = set()
    # Now we can combine the partials to a global average.
    global_sum = 0
    global_count = 0

    for output in results:
        global_sum += output["sum"]
        global_count += output["count"]
        global_features.update(output['features'])

    average = global_sum / global_count

    # Convert the set back to a list
    if not expl_vars:
        expl_vars = list(global_features)

    task = client.create_new_task(
        input_={
            'method': 'corr_partial',
            'kwargs': {'expl_vars': expl_vars, 'average': average,
                       'roitype': roitype}
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

    # Now we combine the partials to a global average.

    sum_prod = 0
    for output in results:
        sum_prod += output["codeviances"]

    sum_prod = sum_prod.reset_index()
    sum_prod.columns = ["Var1", "Var2", "SumProd"]

    var = sum_prod[sum_prod["Var1"] == sum_prod["Var2"]].drop_duplicates()
    std_cols = (var[['Var1', 'SumProd']].set_index("Var1")['SumProd'] / global_count) ** 0.5

    sum_prod = sum_prod.merge(var, on="Var1", suffixes=("", "_v1"))
    sum_prod = sum_prod.merge(var, on="Var2", suffixes=("", "_v2"))

    ## Pearson's coefficient computation
    sum_prod["rho"] = sum_prod["SumProd"] / (sum_prod["SumProd_v1"] * sum_prod["SumProd_v2"]) ** 0.5
    correlationMatrix = pd.pivot_table(sum_prod, values="rho", index="Var1", columns="Var2", aggfunc='mean')

    # correlationMatrix.to_csv(fold + "train_correlationMatrix.csv")

    # corr_ff = correlationMatrix[expl_vars]
    # corr_fc = correlationMatrix[censor_col]
    # best_subset, best_value = CFS(corr_fc, corr_ff)

    return correlationMatrix
    # return {"correlationMatrix": correlationMatrix, 'best_subset': best_subset}


def RPC_average_partial(data, expl_vars, roitype):
    """Compute the average partial

    The data argument in this case contains a dataframe with the RDF endpoint of the node.
    """

    # extract the endpoint from the dataframe; vantage6 3.7.3  assumes csv as input and reads it as dataframe
    if len(data['endpoint']) != 1:
        info('Multiple endpoints defined in data file, using first occurrence')
    endpoint = str(data['endpoint'].iloc[0])

    result_data = extract_corr_via_sparql(roitype, endpoint)
    df = pd.read_csv(StringIO(result_data))
    df = df.drop_duplicates(subset=['subject', 'censor', 'ROI', 'featureName'], keep='first')

    # Get the list of all feature names if not provided
    local_vars = df['featureName'].unique().tolist()
    if len(expl_vars) == 0:
        expl_vars = expl_vars + local_vars

    df = df.pivot(index=['subject', 'censor', 'ROI'], columns='featureName', values='feature_value').fillna(
        0).reset_index()

    local_sum = df[expl_vars].sum()
    local_count = len(df)
    print(local_sum, local_count)
    # return the values as a dict
    return {
        "sum": local_sum,
        "count": local_count,
        "features": local_vars
    }


def RPC_corr_partial(data, expl_vars, average, roitype):
    """Compute the corr partial

    The data argument in this case contains a dataframe with the RDF endpoint of the node.
    """

    # extract the endpoint from the dataframe; vantage6 3.7.3  assumes csv as input and reads it as dataframe
    if len(data['endpoint']) != 1:
        info('Multiple endpoints defined in data file, using first occurrence')
    endpoint = str(data['endpoint'].iloc[0])

    result_data = extract_corr_via_sparql(roitype, endpoint)
    df = pd.read_csv(StringIO(result_data))
    df = df.drop_duplicates(subset=['subject', 'censor', 'ROI', 'featureName'], keep='first')
    df = df.pivot(index=['subject', 'censor', 'ROI'], columns='featureName', values='feature_value').fillna(
        0).reset_index()

    cc = list(product(expl_vars, repeat=2))
    diff = df[expl_vars] - average

    # calculate all sum of products (codeviances)

    df_prod = pd.concat([diff[c[1]] * diff[c[0]] for c in cc], axis=1, keys=cc)
    diff_sum = df_prod.sum()

    print(diff_sum)
    # return the values as a dict
    return {
        "codeviances": diff_sum
    }


def CFS(corr_fc, corr_ff):
    bestMerit = -1
    best_value = -1
    subset = []
    queue = []
    nFails = 0

    # list for visited nodes
    visited = []

    # counter for backtracks
    n_backtrack = 0

    # limit of backtracks
    max_backtrack = 5

    queue = PriorityQueue()
    queue.push(subset, bestMerit)

    # repeat until queue is empty
    # or the maximum number of backtracks is reached
    while not queue.isEmpty():
        # get element of queue with highest merit
        subset, priority = queue.pop()
        # print(subset, priority)

        # check whether the priority of this subset
        # is higher than the current best subset
        if (priority < best_value):
            n_backtrack += 1
        else:
            best_value = priority
            best_subset = subset
            # print(best_value, best_subset)

        # goal condition
        if n_backtrack == max_backtrack:
            break

        # iterate through all feature_type and look of one can
        # increase the merit
        for feature in list(corr_ff):
            temp_subset = subset + [feature]

            # check if this subset has already been evaluated
            for node in visited:
                if set(node) == set(temp_subset):
                    break
            # if not, ...
            else:
                # ... mark it as visited
                visited.append(temp_subset)
                # ... compute merit
                merit = getMerit(temp_subset, corr_fc, corr_ff)  # (temp_subset, label)
                # and push it to the queue
                queue.push(temp_subset, merit)

    return best_subset, best_value


def getMerit(subset, corr_fc, corr_ff):
    k = len(subset)
    rcf = abs(corr_fc[subset]).mean()
    if k > 1:
        sub_corr_ff = corr_ff[subset].loc[subset]
        sub_corr_ff.values[np.tril_indices_from(sub_corr_ff.values)] = np.nan

        sub_corr_ff = abs(sub_corr_ff)
        rff = sub_corr_ff.unstack().mean()
    else:
        rff = 0

    return (k * rcf) / np.sqrt(k + k * (k - 1) * rff)


class PriorityQueue:
    def __init__(self):
        self.queue = []

    def isEmpty(self):
        return len(self.queue) == 0

    def push(self, item, priority):
        """
        item already in priority queue with smaller priority:
        -> update its priority
        item already in priority queue with higher priority:
        -> do nothing
        if item not in priority queue:
        -> push it
        """
        for index, (i, p) in enumerate(self.queue):
            if (set(i) == set(item)):
                if (p >= priority):
                    break
                del self.queue[index]
                self.queue.append((item, priority))
                break
        else:
            self.queue.append((item, priority))

    def pop(self):
        # return item with highest priority and remove it from queue
        max_idx = 0
        for index, (i, p) in enumerate(self.queue):
            if (self.queue[max_idx][1] < p):
                max_idx = index
        (item, priority) = self.queue[max_idx]
        del self.queue[max_idx]
        return (item, priority)


def extract_corr_via_sparql(roitype, endpoint):
    """
    Extracts data via SPARQL of given predicate at specified endpoint

    :param roitype: type of ROI to query - primary or node
    :param str predicate: predicate to extract sum and count of
    :param str endpoint: endpoint to query
    :return: pandas DataFrame containing the local sum and local count
    """
    # define the endpoint with what is available, allowing IP, port, and repo name to be specified from config
    if 'http://' not in endpoint and 'https://' not in endpoint:
        endpoint = f'http://{endpoint}'
    if ':' not in endpoint[endpoint.rfind('://') + 3:]:
        # in case only IP-address is passed
        endpoint = f'{endpoint}:7200/repositories/userRepo'
    if ':' in endpoint[endpoint.rfind('://') + 3:] and '/' not in endpoint[endpoint.rfind('://') + 3:]:
        # in case IP-address and port are passed
        endpoint = f'{endpoint}/repositories/userRepo'

    info(f'Using SPARQL-endpoint: {endpoint}')

    query = compose_sparql_query(roitype)

    # attempt to query
    try:
        annotation_response = requests.post(endpoint, data=f'query={query}',
                                            headers={'Content-Type': 'application/x-www-form-urlencoded'})

        # check whether there is any data with the specified predicate
        response = pd.read_csv(StringIO(annotation_response.text))
        if response.empty:
            info(
                f"Endpoint: {endpoint} does not contain data returning zero to aggregator.")

        response = annotation_response.text
        return response

    except Exception as _exception:
        info(f'Could not query endpoint {endpoint}, response {_exception}')


def compose_sparql_query(roitype):
    """
    Compose a sparql query that gets the feature_type based on the filters specified

    :param list roitype:
    for example as: "GTV-1" or "GTV-2"
    """
    # define prefixes as a string
    prefixes = """ 
            PREFIX ncit: <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#>
            PREFIX roo: <http://www.cancerdata.org/roo/>
            PREFIX ro: <http://www.radiomics.org/RO/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            PREFIX d2rq: <http://www.wiwiss.fu-berlin.de/suhl/bizer/D2RQ/0.1#>
            PREFIX dbo: <http://um-cds/ontologies/databaseontology/>
            PREFIX lidicom: <https://johanvansoest.nl/ontologies/LinkedDicom/>\n"""

    # initialise an empty query string for this input
    query = f"{prefixes}\n"
    query += f"""
        SELECT DISTINCT ?subject ?censor ?ROI ?featureName ?feature_value
        WHERE {{# query feature names and values for the mentioned ROIs
            ?patient a ncit:C16960.
            ?patient roo:P100061 ?Subject_label.
            ?Subject_label dbo:has_cell ?subject_cell.
            ?subject_cell dbo:has_value ?subject.
            ?patient dbo:has_clinical_features ?clinical.
            ?clinical dbo:has_table ?clin_table.
            ?clin_table roo:P100028 ?survival.
            ?survival dbo:has_cell ?survival_cell.
            ?survival_cell dbo:has_value ?censor.
            ?patient roo:has_roi_label ?ROIname.
            ?ROIname dbo:has_cell ?ROIcell.
            ?ROIcell dbo:has_code ?ROIcode.
            ?patient ro:P00088 ?feature.
            ?feature rdf:type ?featureClass.
            ?featureClass dbo:ftype ?featureType.
            ?featureClass dbo:ibsi ?featureName.
            ?feature dbo:has_cell ?feature_cell.
            ?feature_cell roo:P100042 ?feature_value.
            FILTER regex(str(?ROIcode), ("http://www.cancerdata.org/roo/C100346|http://www.cancerdata.org/roo/C100347"))
            BIND(strafter(str(?ROIcode), "http://www.cancerdata.org/roo/") AS ?ROI)
    """

    # filtering the feature_type based on selected ROI (primary or node)
    if roitype:
        filters = ''
        filters = f'FILTER (str(?ROI) = ("{roitype}"))'

        query += f"""
            {filters}
        """

    query += "}"  # closing brace for the initial pattern

    return query
