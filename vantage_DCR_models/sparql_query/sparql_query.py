import time
from vantage6.tools.util import info
import pandas as pd
import requests
from io import StringIO
import os

codedict = {
    "C00000": "Unknown", "C48737": "Tx", "C48719": "T0", "C48720": "T1", "C48724": "T2",
    "C48728": "T3", "C48732": "T4", "C48705": "N0", "C48706": "N1", "C48786": "N2", "C48714": "N3",
    "C28554": 1, "C37987": 0, "C128839": "HPV Positive", "C100346": "Primary", "C100347": "Node",
    "C131488": "HPV Negative", "C10000": "Unknown", "C94626": "Chemotherapy", "C15313": "Radiotherapy",
    "C12762": "Oropharynx", "C12420": "Larynx", "C12246": "Hypopharynx", "C12423": "Nasopharynx",
    "C12421": "Oral cavity", "C150211": "Larynx", "C4044": "Larynx", "C20000": 1, "C30000": 0, "C40000": 1, "C50000": 0}


def master(client, data, expl_vars, time_col, outcome_col, feature_type, organization_ids=None):
    """This package does the following:
            1. Fetch data using SPARQL queries
    """
    path = "/mnt/data/"
    MAX_COMPLEXITY = 250000

    timestamp = str(round(time.time()))
    os.mkdir(timestamp)
    n_covs = len(expl_vars)
    epochs = 10

    # Info messages can help you when an algorithm crashes. These info
    # messages are stored in a log file which is sent to the server when
    # either a task finished or crashes.
    info('Collecting participating organizations')

    # Collect all organization that participate in this collaboration.
    # These organizations will receive the task to compute the partial.
    # organizations = client.get_organizations_in_my_collaboration()
    # ids = [organization.get("id") for organization in organizations]

    if isinstance(organization_ids, list) is False:
        organizations = client.get_organizations_in_my_collaboration()
        ids = [organization.get("id") for organization in organizations]
    else:
        ids = organization_ids

    info(f'sending task to organizations {ids}')

    # print(organizations)
    # Requesting data from the rdf-endpoint (clinical or radiomics or both)
    info('Fetch data using SPARQL queries!')

    if feature_type == "Radiomics" or feature_type == "Combined":
        kwargs_dict = {'expl_vars': expl_vars, 'feature_type': feature_type}
        method = 'average_partial'

        results = subtaskLauncher(client, [method, kwargs_dict, ids])

        # Now we can combine the partials to a global average.
        global_sum = 0
        global_count = 0

        for output in results:
            global_sum += output["sum"]
            global_count += output["count"]

        average = global_sum / global_count

        kwargs_dict = {'expl_vars': expl_vars, 'mean_cols': average, 'feature_type': feature_type}
        method = 'get_std_sums'
        results = subtaskLauncher(client, [method, kwargs_dict, ids])

        # Now we can combine the partials to a global average.

        std_cols = 0
        for output in results:
            std_cols += output["std_col_sums"]

        std_cols = (std_cols / global_count) ** 0.5

        kwargs_dict = {'expl_vars': expl_vars, 'mean_cols': average, 'std_cols': std_cols, 'feature_type': feature_type}
        method = 'normalize'
        results = subtaskLauncher(client, [method, kwargs_dict, ids])
        return results


    else:
        task = client.create_new_task(
            input_={
                'method': 'retrieve_data',
                'kwargs': {
                    'feature_type': feature_type,
                    'expl_vars': expl_vars
                }
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

        for outcome in results:
            info(outcome['fetch'])

        return outcome


def RPC_retrieve_data(data, feature_type, expl_vars):
    path = '/mnt/data/'

    if len(data['endpoint']) != 1:
        info('Multiple endpoints defined in data file, using first occurrence')
    endpoint = str(data['endpoint'].iloc[0])

    result_data = extract_data_via_sparql(endpoint, feature_type, expl_vars)
    df = pd.read_csv(StringIO(result_data))
    for col in df.columns:
        df[col] = df[col].map(codedict).fillna(df[col])

    columns = ["overallsurvivaldays", "metastasisdays", "recurrencedays"]
    statuses = ["survival", "metastasis", "recurrence"]

    for col, status in zip(columns, statuses):
        # follow up to 5 years
        df.loc[df[col] >= 1826, col] = 1826
        df.loc[df[col] >= 1826, status] = 0
        # df.loc[df[col] >= 1095, col] = 1095
        # df.loc[df[col] >= 1095, status] = 0

    TList = ["tstage_Tx", "tstage_T0", "tstage_T1", "tstage_T2", "tstage_T3", "tstage_T4"]
    NList = ["nstage_Nx", "nstage_N0", "nstage_N1", "nstage_N2", "nstage_N3"]

    if feature_type == "Clinical":
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
        df = df.dropna(axis=0)

        columns_to_convert = ['treatment_Radiotherapy', 'treatment_Chemotherapy', 'hpv_HPV Negative', 'hpv_HPV Positive', 'hpv_Unknown']

        for column in columns_to_convert:
            df[column] = df[column].astype(int)

        df.to_csv(path + 'clinical_data.csv', index=False)

    return {'fetch': 'ok'}


def RPC_get_std_sums(data, expl_vars, mean_cols, feature_type):
    # extract the endpoint from the dataframe; vantage6 3.7.3  assumes csv as input and reads it as dataframe
    path = '/mnt/data/'

    if feature_type == "Radiomics":
        file = path + 'radiomics_raw.csv'

    elif feature_type == "Combined":
        file = path + 'combined_raw.csv'

    df = pd.read_csv(file)
    std_col_sums = (df[expl_vars] - mean_cols[expl_vars]) ** 2
    std_col_sums = std_col_sums.sum()
    return {'std_col_sums': std_col_sums}


def RPC_average_partial(data, expl_vars, feature_type):
    """Compute the average partial

    The data argument contains a pandas-dataframe containing the local
    data from the node.
    """
    # extract the endpoint from the dataframe; vantage6 3.7.3  assumes csv as input and reads it as dataframe
    if len(data['endpoint']) != 1:
        info('Multiple endpoints defined in data file, using first occurrence')
    endpoint = str(data['endpoint'].iloc[0])

    result_data = extract_data_via_sparql(endpoint, feature_type, expl_vars)
    df = pd.read_csv(StringIO(result_data))
    # df = data_selector(data, False, data_set)
    for feature in expl_vars:
        df = df[pd.to_numeric(df[feature], errors='coerce').notnull()]
        df[feature] = df[feature].astype(float).round(3)

    path = '/mnt/data/'
    df = df.dropna(axis=0)

    if feature_type == "Radiomics":
        df.to_csv(path + 'radiomics_raw.csv', index=False)

    elif feature_type == "Combined":
        df.to_csv(path + 'combined_raw.csv', index=False)

    local_sum = df[expl_vars].sum()
    local_count = len(df)
    # return the values as a dict
    return {
        "sum": local_sum,
        "count": local_count}


def RPC_normalize(data, expl_vars, mean_cols, std_cols, feature_type):
    path = '/mnt/data/'

    if feature_type == "Radiomics":
        file = path + 'radiomics_raw.csv'

    elif feature_type == "Combined":
        file = path + 'combined_raw.csv'

    df = pd.read_csv(file)
    # Normalize retrieved data
    df[expl_vars] = (df[expl_vars] - mean_cols[expl_vars]) / std_cols[expl_vars]

    for col in df.columns:
        df[col] = df[col].map(codedict).fillna(df[col])

    columns = ["overallsurvivaldays", "metastasisdays", "recurrencedays"]
    statuses = ["survival", "metastasis", "recurrence"]

    for col, status in zip(columns, statuses):
        # follow up to 5 years
        df.loc[df[col] >= 1826, col] = 1826
        df.loc[df[col] >= 1826, status] = 0
        # df.loc[df[col] >= 1095, col] = 1095
        # df.loc[df[col] >= 1095, status] = 0

    columns_to_check = ['patientID', 'ROI', 'survival', 'metastasis', 'recurrence']
    # Exclude the first two columns from the aggregation
    aggregation_cols = [col for col in df.columns if col not in columns_to_check]
    # Group by the first two columns, calculate average for numeric columns, and keep string columns
    df = df.groupby(columns_to_check).agg(
        {col: 'mean' if pd.api.types.is_numeric_dtype(df[col]) else 'first' for col in
         aggregation_cols}).reset_index()
    df = df.dropna(axis=0)

    if feature_type == "Radiomics":
        df.to_csv(path + 'radiomics_data.csv', index=False)

    if feature_type == "Combined":
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

        df = df.dropna(axis=0)

        columns_to_convert = ['treatment_Radiotherapy', 'treatment_Chemotherapy', 'hpv_HPV Negative',
                              'hpv_HPV Positive', 'hpv_Unknown']

        for column in columns_to_convert:
            df[column] = df[column].astype(int)

        df.to_csv(path + 'combined_data.csv', index=False)

    return "Queried data was normalized and saved in mount folder"


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


def extract_data_via_sparql(endpoint, feature_type, expl_vars):
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

    query = compose_sparql_query(feature_type, expl_vars)

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


def compose_sparql_query(feature_type, expl_vars):
    """
    Compose a sparql query that gets the feature_type based on the filters specified

    :param feature_type:
    for example as: "Clinical" or "Radiomics" or "Combined"
    """

    global query
    # define prefixes as a string
    prefixes = """ 
            PREFIX dbo: <http://um-cds/ontologies/databaseontology/>
            PREFIX roo: <http://www.cancerdata.org/roo/>
            PREFIX ncit: <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX ro: <http://www.radiomics.org/RO/>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>\n"""

    if feature_type == 'Clinical':
        # initialise an empty query string for this input
        query = f"{prefixes}\n"
        query += f"""
            SELECT DISTINCT ?patientID ?treatment ?tstage ?nstage ?hpv ?survival ?overallsurvivaldays ?metastasis ?metastasisdays ?recurrence ?recurrencedays ?tumourlocation
            WHERE {{
                ?patient roo:P100061 ?Subject_label.
                ?Subject_label dbo:has_cell ?Subject_cell.
                ?Subject_cell dbo:has_value ?patientID.
            OPTIONAL {{    
                ?patient roo:P100231 ?treatment_type.
                ?treatment_type dbo:has_cell ?treatment_type_cell.
                ?treatment_type_cell a ?treatment_type_class. 
                FILTER regex(str(?treatment_type_class), ("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C94626|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C15313"))
                BIND(strafter(str(?treatment_type_class), "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#") AS ?treatment)
            }}    
            OPTIONAL {{
                ?patient roo:P100029 ?neoplasm.
                ?neoplasm roo:P100244 ?t_type.
                ?t_type dbo:has_cell ?t_type_cell.
                ?t_type_cell a ?t_type_class.
                FILTER regex(str(?t_type_class), ("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48737|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48719|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48720|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48724|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48728|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48732|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48733|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48734|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C132010")) 
                BIND(strafter(str(?t_type_class), "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#") AS ?tstage)
            }}
            OPTIONAL {{    
                ?patient roo:P100029 ?neoplasm.
                ?neoplasm roo:P100242 ?n_type.
                ?n_type dbo:has_cell ?n_type_cell.
                ?n_type_cell a ?n_type_class.
                FILTER regex(str(?n_type_class), ("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48705|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48706|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48786|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48711|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48712|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48713|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48714|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48715|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48716|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C10000"))
                BIND(strafter(str(?n_type_class), "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#") AS ?nstage)
            }}
            OPTIONAL {{
                ?patient roo:P100022 ?hpv_type.
                ?hpv_type dbo:has_cell ?hpv_cell.
                ?hpv_cell a ?hpv_class.
                FILTER regex(str(?hpv_class), ("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C128839|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C131488|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C10000"))
                BIND(strafter(str(?hpv_class), "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#") AS ?hpv)
            }}
            OPTIONAL {{
                ?patient roo:P100028 ?surv_type.
                ?surv_type dbo:has_cell ?surv_cell.
                ?surv_cell a ?surv_class.
                FILTER regex(str(?surv_class), ("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C28554|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C37987"))
                BIND(strafter(str(?surv_class), "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#") AS ?survival)   
            }}
                ?patient roo:P100026 ?overallsurvival.
                ?overallsurvival dbo:has_cell ?overallsurvival_cell.
                ?overallsurvival_cell dbo:has_value ?overallsurvivaldays.
                
            OPTIONAL {{
                ?patient roo:P100029 ?neoplasm.
                ?neoplasm roo:P10032 ?meta_type.
                ?meta_type dbo:has_cell ?meta_cell.
                ?meta_cell a ?meta_cell_class.
                FILTER regex(str(?meta_cell_class), ("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C20000|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C30000"))
                BIND(strafter(str(?meta_cell_class), "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#") AS ?metastasis)   
                #?meta_cell dbo:has_value ?metastasis.  
            }} 

            OPTIONAL {{
                ?patient roo:P100029 ?neoplasm.
                ?neoplasm roo:P100022 ?meta_days.
                ?meta_days a ?meta_days_class.
                ?meta_days_class owl:equivalentClass roo:metastasisdays.
                ?meta_days dbo:has_cell ?meta_days_cell.
                ?meta_days_cell dbo:has_value ?metastasisdays. 
            }}
            
            OPTIONAL {{
                ?patient roo:P100029 ?neoplasm.
                ?neoplasm roo:P100022 ?lrr.
                ?lrr a ?lrr_class.
                ?lrr_class owl:equivalentClass roo:regionalrecurrence.
                ?lrr dbo:has_cell ?lrr_cell.
        		?lrr_cell a ?lrr_cell_class.
                FILTER regex(str(?lrr_cell_class), ("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C40000|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C50000")).
                BIND(strafter(str(?lrr_cell_class), "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#") AS ?recurrence).   
                #?lrr_cell dbo:has_value ?recurrence.  
            }} 

            OPTIONAL {{
                ?patient roo:P100029 ?neoplasm.
                ?neoplasm roo:P100022 ?lrr_days.
                ?lrr_days a ?lrr_days_class.
                ?lrr_days_class owl:equivalentClass roo:regionalrecurrencedays.
                ?lrr_days dbo:has_cell ?lrr_days_cell.
                ?lrr_days_cell dbo:has_value ?recurrencedays.
            }}

            ?patient roo:P100029 ?neoplasm.
            ?neoplasm roo:P100202 ?tumour.
            ?tumour dbo:has_cell ?tumour_cell.
            ?tumour_cell a ?tumour_class.
            FILTER regex(str(?tumour_class), ("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C12762|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C12246|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C12420|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C12421|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C4044|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C150211|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C12423|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C00000"))
            BIND(strafter(str(?tumour_class), "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#") AS ?tumourlocation).   
        }}    
        """

    elif feature_type == 'Radiomics':
        query = f"{prefixes}\n"
        query += f"""
            SELECT DISTINCT ?patientID ?ROI ?survival ?overallsurvivaldays ?metastasis ?metastasisdays ?recurrence ?recurrencedays ?tumourlocation
            """

        # Iterate through the list of features and add the block for each feature in select statement
        for feature in expl_vars:
            query += f"?{feature} "

        # Add the WHERE clause
        query += """
            WHERE {
            """
        # Iterate through the list of features and add the block for each feature
        for feature in expl_vars:
            # Extract the last word of the feature
            replaced_feature = feature.replace("_", ".")

            # Add the block for the current feature
            query += f"""
                ?patient ro:P00088 ?feature_{feature}.
                ?feature_{feature} a ?feature_type_{feature}.
                ?feature_type_{feature} dbo:ibsi "{replaced_feature}".
                ?feature_{feature} dbo:has_cell ?feature_{feature}_cell.
                ?feature_{feature}_cell dbo:has_value ?{feature}.
                """

        query += f"""

            ?patient roo:P100061 ?Subject_label.
            ?Subject_label dbo:has_cell ?Subject_cell.
            ?Subject_cell dbo:has_value ?patientID.

            ?patient roo:has_roi_label ?ROIname.
            ?ROIname dbo:has_cell ?ROIcell.
            ?ROIcell dbo:has_code ?ROIcode.
            FILTER regex(str(?ROIcode), ("http://www.cancerdata.org/roo/C100346|http://www.cancerdata.org/roo/C100347"))
            BIND(strafter(str(?ROIcode), "http://www.cancerdata.org/roo/") AS ?ROI)

            ?patient dbo:has_clinical_features ?clin_table.
            ?clin_table dbo:has_table ?clinical.
            ?clinical roo:P100028 ?surv_type.
            ?surv_type dbo:has_cell ?surv_cell.
            ?surv_cell a ?surv_class.
            FILTER regex(str(?surv_class), ("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C28554|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C37987"))
            BIND(strafter(str(?surv_class), "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#") AS ?survival)  

            ?clinical roo:P100026 ?overallsurvival.
            ?overallsurvival dbo:has_cell ?overallsurvival_cell.
            ?overallsurvival_cell dbo:has_value ?overallsurvivaldays
            
            OPTIONAL {{
                ?clinical roo:P100029 ?neoplasm.
                ?neoplasm roo:P10032 ?meta_type.
                ?meta_type dbo:has_cell ?meta_cell.
                ?meta_cell a ?meta_cell_class.
                FILTER regex(str(?meta_cell_class), ("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C20000|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C30000"))
                BIND(strafter(str(?meta_cell_class), "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#") AS ?metastasis)   
                #?meta_cell dbo:has_value ?metastasis.  
            }} 

            OPTIONAL {{
                ?clinical roo:P100029 ?neoplasm.
                ?neoplasm roo:P100022 ?meta_days.
                ?meta_days a ?meta_days_class.
                ?meta_days_class owl:equivalentClass roo:metastasisdays.
                ?meta_days dbo:has_cell ?meta_days_cell.
                ?meta_days_cell dbo:has_value ?metastasisdays. 
            }}
            
            OPTIONAL {{
                ?clinical roo:P100029 ?neoplasm.
                ?neoplasm roo:P100022 ?lrr.
                ?lrr a ?lrr_class.
                ?lrr_class owl:equivalentClass roo:regionalrecurrence.
                ?lrr dbo:has_cell ?lrr_cell.
        		?lrr_cell a ?lrr_cell_class.
                FILTER regex(str(?lrr_cell_class), ("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C40000|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C50000")).
                BIND(strafter(str(?lrr_cell_class), "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#") AS ?recurrence).   
                #?lrr_cell dbo:has_value ?recurrence.  
            }} 

            OPTIONAL {{
                ?clinical roo:P100029 ?neoplasm.
                ?neoplasm roo:P100022 ?lrr_days.
                ?lrr_days a ?lrr_days_class.
                ?lrr_days_class owl:equivalentClass roo:regionalrecurrencedays.
                ?lrr_days dbo:has_cell ?lrr_days_cell.
                ?lrr_days_cell dbo:has_value ?recurrencedays.
            }}

            ?clinical roo:P100029 ?neoplasm.
            ?neoplasm roo:P100202 ?tumour.
            ?tumour dbo:has_cell ?tumour_cell.
            ?tumour_cell a ?tumour_class.
            FILTER regex(str(?tumour_class), ("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C12762|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C12246|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C12420|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C12421|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C4044|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C150211|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C12423|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C00000"))
            BIND(strafter(str(?tumour_class), "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#") AS ?tumourlocation)   
            #FILTER (str(?tumourlocation) != ("C12762"))

            }}
            """

    elif feature_type == 'Combined':
        query = f"{prefixes}\n"
        query += f"""
            SELECT DISTINCT ?patientID ?ROI ?survival ?overallsurvivaldays ?metastasis ?metastasisdays ?recurrence ?recurrencedays ?tumourlocation ?treatment ?tstage ?nstage ?hpv
            """
        # Iterate through the list of features and add the block for each feature in select statement
        for feature in expl_vars:
            query += f"?{feature} "

            # Add the WHERE clause
        query += """
                    WHERE {
                    """
        # Iterate through the list of features and add the block for each feature
        for feature in expl_vars:
            # Extract the last word of the feature
            replaced_feature = feature.replace("_", ".")

            # Add the block for the current feature
            query += f"""
                        ?patient ro:P00088 ?feature_{feature}.
                        ?feature_{feature} a ?feature_type_{feature}.
                        ?feature_type_{feature} dbo:ibsi "{replaced_feature}".
                        ?feature_{feature} dbo:has_cell ?feature_{feature}_cell.
                        ?feature_{feature}_cell dbo:has_value ?{feature}.
                        """

        query += f"""

            ?patient roo:P100061 ?Subject_label.
            ?Subject_label dbo:has_cell ?Subject_cell.
            ?Subject_cell dbo:has_value ?patientID.

            ?patient roo:has_roi_label ?ROIname.
            ?ROIname dbo:has_cell ?ROIcell.
            ?ROIcell dbo:has_code ?ROIcode.
            FILTER regex(str(?ROIcode), ("http://www.cancerdata.org/roo/C100346|http://www.cancerdata.org/roo/C100347"))
            BIND(strafter(str(?ROIcode), "http://www.cancerdata.org/roo/") AS ?ROI)

            ?patient dbo:has_clinical_features ?clin_table.
            ?clin_table dbo:has_table ?clinical.
            
            OPTIONAL {{
                ?clinical roo:P100231 ?treatment_type.
                ?treatment_type dbo:has_cell ?treatment_type_cell.
                ?treatment_type_cell a ?treatment_type_class. 
                FILTER regex(str(?treatment_type_class), ("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C94626|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C15313"))
                BIND(strafter(str(?treatment_type_class), "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#") AS ?treatment)
            }}
            
            OPTIONAl {{
                ?clinical roo:P100029 ?neoplasm.
                ?neoplasm roo:P100244 ?t_type.
                ?t_type dbo:has_cell ?t_type_cell.
                ?t_type_cell a ?t_type_class.
                FILTER regex(str(?t_type_class), ("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48737|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48719|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48720|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48724|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48728|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48732|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48733|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48734"))
                BIND(strafter(str(?t_type_class), "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#") AS ?tstage)
            }}

            OPTIONAL {{
                ?clinical roo:P100029 ?neoplasm.
                ?neoplasm roo:P100242 ?n_type.
                ?n_type dbo:has_cell ?n_type_cell.
                ?n_type_cell a ?n_type_class.
                FILTER regex(str(?n_type_class), ("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48705|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48706|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48786|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48711|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48712|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48713|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48714|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48715|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48716"))
                BIND(strafter(str(?n_type_class), "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#") AS ?nstage)
            }} 

            OPTIONAL {{
                ?clinical roo:P100022 ?hpv_type.
                ?hpv_type dbo:has_cell ?hpv_cell.
                ?hpv_cell a ?hpv_class.
                FILTER regex(str(?hpv_class), ("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C128839|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C131488|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C10000"))
                BIND(strafter(str(?hpv_class), "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#") AS ?hpv)
            }}
            
            ?clinical roo:P100028 ?surv_type.
            ?surv_type dbo:has_cell ?surv_cell.
            ?surv_cell a ?surv_class.
            FILTER regex(str(?surv_class), ("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C28554|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C37987"))
            BIND(strafter(str(?surv_class), "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#") AS ?survival)  

            ?clinical roo:P100026 ?overallsurvival.
            ?overallsurvival dbo:has_cell ?overallsurvival_cell.
            ?overallsurvival_cell dbo:has_value ?overallsurvivaldays

            OPTIONAL {{
                ?clinical roo:P100029 ?neoplasm.
                ?neoplasm roo:P10032 ?meta_type.
                ?meta_type dbo:has_cell ?meta_cell.
                ?meta_cell a ?meta_cell_class.
                FILTER regex(str(?meta_cell_class), ("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C20000|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C30000"))
                BIND(strafter(str(?meta_cell_class), "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#") AS ?metastasis)   
                #?meta_cell dbo:has_value ?metastasis.  
            }} 

            OPTIONAL {{
                ?clinical roo:P100029 ?neoplasm.
                ?neoplasm roo:P100022 ?meta_days.
                ?meta_days a ?meta_days_class.
                ?meta_days_class owl:equivalentClass roo:metastasisdays.
                ?meta_days dbo:has_cell ?meta_days_cell.
                ?meta_days_cell dbo:has_value ?metastasisdays. 
            }}
            
            OPTIONAL {{
                ?clinical roo:P100029 ?neoplasm.
                ?neoplasm roo:P100022 ?lrr.
                ?lrr a ?lrr_class.
                ?lrr_class owl:equivalentClass roo:regionalrecurrence.
                ?lrr dbo:has_cell ?lrr_cell.
        		?lrr_cell a ?lrr_cell_class.
                FILTER regex(str(?lrr_cell_class), ("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C40000|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C50000")).
                BIND(strafter(str(?lrr_cell_class), "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#") AS ?recurrence).   
                #?lrr_cell dbo:has_value ?recurrence.  
            }} 

            OPTIONAL {{
                ?clinical roo:P100029 ?neoplasm.
                ?neoplasm roo:P100022 ?lrr_days.
                ?lrr_days a ?lrr_days_class.
                ?lrr_days_class owl:equivalentClass roo:regionalrecurrencedays.
                ?lrr_days dbo:has_cell ?lrr_days_cell.
                ?lrr_days_cell dbo:has_value ?recurrencedays.
            }}

            ?clinical roo:P100029 ?neoplasm.
            ?neoplasm roo:P100202 ?tumour.
            ?tumour dbo:has_cell ?tumour_cell.
            ?tumour_cell a ?tumour_class.
            FILTER regex(str(?tumour_class), ("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C12762|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C12246|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C12420|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C12421|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C4044|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C150211|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C12423|http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C00000"))
            BIND(strafter(str(?tumour_class), "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#") AS ?tumourlocation)   
            #FILTER (str(?tumourlocation) != ("C12762"))

            }}
            """
    else:
        info(f'data-set type not valid: {feature_type}. Mention "Combined", "Clinical" or "Radiomics"')

    return query
