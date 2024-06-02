import requests
import time
from itertools import product
import numpy as np
import pandas as pd

from vantage6.tools.util import info
from io import StringIO

codedict = {
    "C00000": "Unknown", "C48737": "Tx", "C48719": "T0", "C48720": "T1", "C48724": "T2",
    "C48728": "T3", "C48732": "T4", "C48705": "N0", "C48706": "N1", "C48786": "N2", "C48714": "N3",
    "C28554": 1, "C37987": 0, "C128839": "HPV Positive", "C100346": "Primary", "C100347": "Node",
    "C131488": "HPV Negative", "C10000": "Unknown", "C94626": "Chemotherapy", "C15313": "Radiotherapy",
    "C12762": "Oropharynx", "C12420": "Larynx", "C12246": "Hypopharynx", "C12423": "Nasopharynx",
    "C12421": "Oral cavity", "C20000": 1, "C30000": 0, "C40000": 1, "C50000": 0}


def master(client, data, roitype, expl_vars, outcome_col, oropharynx, feature_type, organization_ids=None):
    """
    Combine partials to global model for the provided predicate and specified organisations

    :param client: vantage client
    :param data: data defined in node config file
    :param roitype: roi type to retrieve the average from, defined by user (Primary or Node or Combined)
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
            'kwargs': {'expl_vars': expl_vars + [outcome_col], 'feature_type': feature_type,
                       'roitype': roitype, 'oropharynx': oropharynx}
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

    kwargs_dict = {'expl_vars': expl_vars + [outcome_col], 'mean_cols': average}
    method = 'get_std_sums'
    results = subtaskLauncher(client, [method, kwargs_dict, ids])

    # Now we can combine the partials to a global average.

    std_cols = 0
    for output in results:
        std_cols += output["std_col_sums"]

    std_cols = (std_cols / global_count) ** 0.5

    kwargs_dict = {'expl_vars': expl_vars, 'mean_cols': average, 'std_cols': std_cols}
    method = 'normalize'
    results = subtaskLauncher(client, [method, kwargs_dict, ids])
    print(results)

    task = client.create_new_task(
        input_={
            'method': 'corr_partial',
            'kwargs': {'expl_vars': expl_vars + [outcome_col], 'average': average,
                       'roitype': roitype, 'oropharynx': oropharynx}
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

    corr_ff = correlationMatrix[expl_vars]
    corr_fc = correlationMatrix[outcome_col]
    best_subset, best_value = CFS(corr_fc, corr_ff)

    info('save results')
    print(correlationMatrix)
    task = client.create_new_task(
        input_={
            'method': 'save_results',
            'kwargs': {'correlationMatrix': correlationMatrix, 'roitype': roitype
                , 'best_subset': best_subset, 'average': average, 'std_cols': std_cols, 'global_count': global_count}
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

    # return correlationMatrix
    return {"correlationMatrix": correlationMatrix, 'best_subset': best_subset}


def RPC_save_results(data, correlationMatrix, best_subset, average, std_cols, global_count, roitype):
    fold = '/mnt/data/'
    # average.to_csv(fold + f'{roitype}_means.csv')
    # pd.DataFrame(np.array([global_count]), columns=['tot']).to_csv(fold + f'{roitype}_total_patients.csv')
    correlationMatrix.to_csv(fold + f'{roitype}_correlationMatrix.csv')
    pd.DataFrame(best_subset, columns=["sel_expl_vars"]).to_csv(fold + f'{roitype}_sel_expl_vars.csv')
    # std_cols.to_csv(fold + f'{roitype}_std.csv')
    return 'ok'


def RPC_average_partial(data, expl_vars, feature_type, roitype, oropharynx):
    """Compute the average partial

    The data argument in this case contains a dataframe with the RDF endpoint of the node.
    """
    # extract the endpoint from the dataframe; vantage6 3.7.3  assumes csv as input and reads it as dataframe
    if len(data['endpoint']) != 1:
        info('Multiple endpoints defined in data file, using first occurrence')
    endpoint = str(data['endpoint'].iloc[0])

    result_data = extract_corr_via_sparql(roitype, feature_type, endpoint, oropharynx)
    df = pd.read_csv(StringIO(result_data))
    df = df.drop(['tumourlocation'], axis=1)
    for col in df.columns:
        df[col] = df[col].map(codedict).fillna(df[col])

    if feature_type == 'Radiomics':

        df = df.drop_duplicates(subset=['subject', 'survival', 'metastasis', 'recurrence', 'ROI', 'featureName'],
                                keep='first')
        local_vars = df['featureName'].unique().tolist()
        if len(expl_vars) == 1:
            expl_vars = expl_vars + local_vars

        df = df[pd.to_numeric(df['feature_value'], errors='coerce').notnull()]
        df["feature_value"] = df["feature_value"].astype(float).round(3)

        if roitype != "Combined":
            df = df.pivot(index=['subject', 'survival', 'metastasis', 'recurrence', 'ROI'], columns='featureName',
                          values='feature_value').fillna(0).reset_index()
        else:
            df = df.pivot(index=['subject', 'survival', 'metastasis', 'recurrence'], columns=['ROI', 'featureName'],
                          values='feature_value').fillna(0).reset_index()
            # Flatten the columns
            df.columns = ['{}_{}'.format(col[0], col[1]) for col in df.columns]
            df.columns = df.columns.str.rstrip('_')

    elif feature_type == 'Combined':

        df = df.drop_duplicates(subset=['subject', 'survival', 'metastasis', 'recurrence', 'treatment', 'tstage', 'nstage', 'hpv', 'ROI', 'featureName'],
                                keep='first')
        local_vars = df['featureName'].unique().tolist()
        local_vars = local_vars + ['T2orLower', 'T3orHigher', 'N1orLower', 'N2orHigher', 'treatment_Chemotherapy', 'treatment_Radiotherapy', 'hpv_HPV Negative', 'hpv_HPV Positive', 'hpv_Unknown']
        if len(expl_vars) == 1:
            expl_vars = expl_vars + local_vars

        df = df[pd.to_numeric(df['feature_value'], errors='coerce').notnull()]
        df["feature_value"] = df["feature_value"].astype(float).round(3)

        if roitype != "Combined":
            df = df.pivot(index=['subject', 'survival', 'metastasis', 'recurrence', 'treatment', 'tstage', 'nstage', 'hpv', 'ROI'], columns='featureName',
                          values='feature_value').fillna(0).reset_index()
        else:
            df = df.pivot(index=['subject', 'survival', 'metastasis', 'recurrence', 'treatment', 'tstage', 'nstage', 'hpv',], columns=['ROI', 'featureName'],
                          values='feature_value').fillna(0).reset_index()
            # Flatten the columns
            df.columns = ['{}_{}'.format(col[0], col[1]) for col in df.columns]
            df.columns = df.columns.str.rstrip('_')

        TList = ["tstage_Tx", "tstage_T0", "tstage_T1", "tstage_T2", "tstage_T3", "tstage_T4"]
        NList = ["nstage_Nx", "nstage_N0", "nstage_N1", "nstage_N2", "nstage_N3"]

        df = pd.get_dummies(df, columns=['treatment', 'tstage', 'hpv', 'nstage'])

        for i in TList:
            if i not in df:
                df[i] = 0
        for j in NList:
            if j not in df:
                df[j] = 0
        df['T2orLower'] = (df['tstage_T0'] | df['tstage_T1'] | df['tstage_T2']).astype(int)
        df['T3orHigher'] = (df['tstage_T3'] | df['tstage_T4']).astype(int)
        df['N1orLower'] = (df['nstage_N0'] | df['nstage_N1']).astype(int)
        df['N2orHigher'] = (df['nstage_N2'] | df['nstage_N3']).astype(int)

    fold = '/mnt/data/'
    df = df.dropna(axis=0)
    df.to_csv(fold + 'corr_data_rad.csv', index=False)

    local_sum = df[expl_vars].sum()
    local_count = len(df)
    print(local_sum, local_count)
    # return the values as a dict
    return {
        "sum": local_sum,
        "count": local_count,
        "features": local_vars
    }


def RPC_get_std_sums(data, expl_vars, mean_cols):
    fold = '/mnt/data/'
    file = fold + 'corr_data_rad.csv'
    df = pd.read_csv(file)
    std_col_sums = (df[expl_vars] - mean_cols[expl_vars]) ** 2
    std_col_sums = std_col_sums.sum()
    return {'std_col_sums': std_col_sums}


def RPC_normalize(data, expl_vars, mean_cols, std_cols):
    fold = '/mnt/data/'
    # nomalize the training set
    file = fold + 'corr_data_rad.csv'
    df = pd.read_csv(file)
    df[expl_vars] = (df[expl_vars] - mean_cols[expl_vars]) / std_cols[expl_vars]
    df.to_csv(fold + 'norm_corr_data_rad.csv', index=False)


def RPC_corr_partial(data, expl_vars, average, roitype, oropharynx):
    """Compute the corr partial

    The data argument in this case contains a dataframe with the RDF endpoint of the node.
    """

    # extract the endpoint from the dataframe; vantage6 3.7.3  assumes csv as input and reads it as dataframe
    fold = '/mnt/data/'
    file = fold + 'norm_corr_data_rad.csv'
    df = pd.read_csv(file)

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


def extract_corr_via_sparql(roitype, feature_type, endpoint, oropharynx):
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

    query = compose_sparql_query(roitype, feature_type, oropharynx)

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


def compose_sparql_query(roitype, feature_type, oropharynx):
    """
    Compose a sparql query that gets the feature_type based on the filters specified

    :param list roitype:
    for example as: "GTV-1" or "GTV-2"
    """
    if feature_type == 'Radiomics':
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
            SELECT DISTINCT ?subject ?survival ?metastasis ?recurrence ?tumourlocation ?ROI ?featureName ?feature_value
            WHERE {{# query feature names and values for the mentioned ROIs
                ?patient a ncit:C16960.
                ?patient roo:P100061 ?Subject_label.
                ?Subject_label dbo:has_cell ?subject_cell.
                ?subject_cell dbo:has_value ?subject.
                ?patient dbo:has_clinical_features ?clinical.
                ?clinical dbo:has_table ?clin_table.
                ?clin_table roo:P100028 ?survival_type.
                ?survival_type dbo:has_cell ?survival_cell.
                ?survival_cell a ?surv_class.
                FILTER regex(str(?surv_class), ("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C28554|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C37987"))
                BIND(strafter(str(?surv_class), "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#") AS ?survival) 
        
                ?clin_table roo:P100029 ?neoplasm.
                ?neoplasm roo:P10032 ?meta_type.
                ?meta_type dbo:has_cell ?meta_cell.
                ?meta_cell a ?meta_cell_class.
                FILTER regex(str(?meta_cell_class), ("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C20000|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C30000"))
                BIND(strafter(str(?meta_cell_class), "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#") AS ?metastasis)   
                #?meta_cell dbo:has_value ?metastasis.  
        
                ?clin_table roo:P100029 ?neoplasm.
                ?neoplasm roo:P100022 ?lrr.
                ?lrr a ?lrr_class.
                ?lrr_class owl:equivalentClass roo:regionalrecurrence.
                ?lrr dbo:has_cell ?lrr_cell.
                ?lrr_cell a ?lrr_cell_class.
                FILTER regex(str(?lrr_cell_class), ("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C40000|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C50000")).
                BIND(strafter(str(?lrr_cell_class), "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#") AS ?recurrence).   
                #?lrr_cell dbo:has_value ?recurrence. 
                
                ?clin_table roo:P100029 ?neoplasm.
                ?neoplasm roo:P100202 ?tumour.
                ?tumour dbo:has_cell ?tumour_cell.
                ?tumour_cell a ?tumour_class.
                FILTER regex(str(?tumour_class), ("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C12762|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C12246|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C12420|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C12421|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C4044|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C150211|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C12423|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C00000"))
                BIND(strafter(str(?tumour_class), "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#") AS ?tumourlocation)    
                
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
        if roitype == "Primary":
            filters = f'FILTER (str(?ROI) = ("C100346"))'
            query += f"""
                {filters}
            """
        elif roitype == "Node":
            filters = f'FILTER (str(?ROI) != ("C100346"))'
            query += f"""
                {filters}
            """
        if oropharynx == "yes":
            filters = f'FILTER (str(?tumourlocation) = ("C12762"))'
            query += f"""
                {filters}
            """
        elif oropharynx == "no":
            filters = f'FILTER (str(?tumourlocation) != ("C12762"))'
            query += f"""
                {filters}
            """

        query += "}"  # closing brace for the initial pattern

    elif feature_type == "Combined":

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
                    SELECT DISTINCT ?subject ?survival ?metastasis ?recurrence ?tumourlocation ?treatment ?tstage ?nstage ?hpv ?ROI ?featureName ?feature_value
                    WHERE {{# query feature names and values for the mentioned ROIs
                        ?patient a ncit:C16960.
                        ?patient roo:P100061 ?Subject_label.
                        ?Subject_label dbo:has_cell ?subject_cell.
                        ?subject_cell dbo:has_value ?subject.
                        ?patient dbo:has_clinical_features ?clinical.
                        ?clinical dbo:has_table ?clin_table.
                        ?clin_table roo:P100028 ?survival_type.
                        ?survival_type dbo:has_cell ?survival_cell.
                        ?survival_cell a ?surv_class.
                        FILTER regex(str(?surv_class), ("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C28554|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C37987"))
                        BIND(strafter(str(?surv_class), "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#") AS ?survival) 

                        ?clin_table roo:P100029 ?neoplasm.
                        ?neoplasm roo:P10032 ?meta_type.
                        ?meta_type dbo:has_cell ?meta_cell.
                        ?meta_cell a ?meta_cell_class.
                        FILTER regex(str(?meta_cell_class), ("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C20000|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C30000"))
                        BIND(strafter(str(?meta_cell_class), "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#") AS ?metastasis)   
                        #?meta_cell dbo:has_value ?metastasis.  

                        ?clin_table roo:P100029 ?neoplasm.
                        ?neoplasm roo:P100022 ?lrr.
                        ?lrr a ?lrr_class.
                        ?lrr_class owl:equivalentClass roo:regionalrecurrence.
                        ?lrr dbo:has_cell ?lrr_cell.
                        ?lrr_cell a ?lrr_cell_class.
                        FILTER regex(str(?lrr_cell_class), ("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C40000|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C50000")).
                        BIND(strafter(str(?lrr_cell_class), "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#") AS ?recurrence).   
                        #?lrr_cell dbo:has_value ?recurrence. 

                        ?clin_table roo:P100029 ?neoplasm.
                        ?neoplasm roo:P100202 ?tumour.
                        ?tumour dbo:has_cell ?tumour_cell.
                        ?tumour_cell a ?tumour_class.
                        FILTER regex(str(?tumour_class), ("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C12762|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C12246|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C12420|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C12421|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C4044|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C150211|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C12423|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C00000"))
                        BIND(strafter(str(?tumour_class), "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#") AS ?tumourlocation) 
                        
                        OPTIONAL {{
                            ?clin_table roo:P100231 ?treatment_type.
                            ?treatment_type dbo:has_cell ?treatment_type_cell.
                            ?treatment_type_cell a ?treatment_type_class. 
                            FILTER regex(str(?treatment_type_class), ("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C94626|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C15313"))
                            BIND(strafter(str(?treatment_type_class), "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#") AS ?treatment)
                        }}
                        
                        OPTIONAl {{
                            ?clin_table roo:P100029 ?neoplasm.
                            ?neoplasm roo:P100244 ?t_type.
                            ?t_type dbo:has_cell ?t_type_cell.
                            ?t_type_cell a ?t_type_class.
                            FILTER regex(str(?t_type_class), ("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48737|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48719|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48720|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48724|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48728|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48732|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48733|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48734"))
                            BIND(strafter(str(?t_type_class), "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#") AS ?tstage)
                        }}
            
                        OPTIONAL {{
                            ?clin_table roo:P100029 ?neoplasm.
                            ?neoplasm roo:P100242 ?n_type.
                            ?n_type dbo:has_cell ?n_type_cell.
                            ?n_type_cell a ?n_type_class.
                            FILTER regex(str(?n_type_class), ("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48705|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48706|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48786|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48711|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48712|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48713|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48714|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48715|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48716"))
                            BIND(strafter(str(?n_type_class), "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#") AS ?nstage)
                        }} 
            
                        OPTIONAL {{
                            ?clin_table roo:P100022 ?hpv_type.
                            ?hpv_type dbo:has_cell ?hpv_cell.
                            ?hpv_cell a ?hpv_class.
                            FILTER regex(str(?hpv_class), ("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C128839|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C131488|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C10000"))
                            BIND(strafter(str(?hpv_class), "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#") AS ?hpv)
                        }}   

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
        if roitype == "Primary":
            filters = f'FILTER (str(?ROI) = ("C100346"))'
            query += f"""
                        {filters}
                    """
        elif roitype == "Node":
            filters = f'FILTER (str(?ROI) != ("C100346"))'
            query += f"""
                        {filters}
                    """
        if oropharynx == "yes":
            filters = f'FILTER (str(?tumourlocation) = ("C12762"))'
            query += f"""
                        {filters}
                    """
        elif oropharynx == "no":
            filters = f'FILTER (str(?tumourlocation) != ("C12762"))'
            query += f"""
                        {filters}
                    """

        query += "}"  # closing brace for the initial pattern

    return query

