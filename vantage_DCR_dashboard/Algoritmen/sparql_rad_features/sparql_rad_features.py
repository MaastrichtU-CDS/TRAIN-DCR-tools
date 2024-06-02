import time
import requests
import pandas as pd
import numpy as np
from vantage6.tools.util import info
from io import StringIO


def master(client, data, feature, organization_ids=None):
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

    # Request all (specified) organisations parties to compute their partial
    info('Requesting partial average using sparql')
    task = client.create_new_task(
        input_={
            'method': 'features_sparql_partial',
            'kwargs': {'feature': feature}
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

    return results


def RPC_features_sparql_partial(data, feature):
    """
    Compute the average partial
    The data argument is supposed to contain a txt file with the ip address of the SPARQL-endpoint.

    :param data: define the SPARQL-endpoint in a csv file under column 'endpoint'; will read first row
    :param str predicate: name of the column to take the average of
    :return: local sum and local count
    """
    # extract the predicate from the dataframe.
    info(f'Extracting features')

    # extract the endpoint from the dataframe; vantage6 3.7.3  assumes csv as input and reads it as dataframe
    if len(data['endpoint']) != 1:
        info('Multiple endpoints defined in data file, using first occurrence')
    endpoint = str(data['endpoint'].iloc[0])

    # extract data and calculate at the sparql endpoint to ensure no data crosses over
    df = extract_features_via_sparql(endpoint, feature)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    stats = boxplot_stats(df)

    # return the values as a dictionary
    return stats


def boxplot_stats(df):
    stats = {}
    for column in df.columns:
        col_data = df[column].dropna()  # Drop NaN values
        minimum = col_data.min()
        q1 = col_data.quantile(0.25)
        median = col_data.median()
        q3 = col_data.quantile(0.75)
        maximum = col_data.max()
        iqr = q3 - q1
        lower_whisker = max(minimum, q1 - 1.5 * iqr)
        upper_whisker = min(maximum, q3 + 1.5 * iqr)
        outliers = col_data[(col_data < lower_whisker) | (col_data > upper_whisker)]

        stats = {
            'min': minimum,
            'q1': q1,
            'median': median,
            'q3': q3,
            'max': maximum,
            'lower_whisker': lower_whisker,
            'upper_whisker': upper_whisker,
            'outliers': outliers.tolist()
        }

    return stats


def extract_features_via_sparql(endpoint, feature):
    """
    Extracts data via SPARQL of given predicate at specified endpoint

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

    # formulate query
    global query
    # define prefixes as a string
    prefixes = """ 
                PREFIX dbo: <http://um-cds/ontologies/databaseontology/>
                PREFIX roo: <http://www.cancerdata.org/roo/>
                PREFIX ncit: <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#>
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX ro: <http://www.radiomics.org/RO/>
                PREFIX owl: <http://www.w3.org/2002/07/owl#>\n"""

    # replace the dot in the feature name with an underscore
    replaced_feature = feature.replace(".", "_")

    query = f"{prefixes}\n"
    query += f"""
                SELECT ?{replaced_feature} 
                WHERE {{
                """

    # Add the block for the current feature
    query += f"""
        ?patient ro:P00088 ?feature_{replaced_feature}.
        ?feature_{replaced_feature} a ?feature_type_{replaced_feature}.
        ?feature_type_{replaced_feature} dbo:ibsi "{feature}".
        ?feature_{replaced_feature} dbo:has_cell ?feature_{replaced_feature}_cell.
        ?feature_{replaced_feature}_cell dbo:has_value ?{replaced_feature}.
        }}
        """

    # attempt to query
    try:
        annotation_response = requests.post(endpoint, data=f'query={query}',
                                            headers={'Content-Type': 'application/x-www-form-urlencoded'})

        # check whether there is any data with the specified predicate
        response = pd.read_csv(StringIO(annotation_response.text))

        if response.empty:
            info(
                f"Endpoint: {endpoint} does not contain data for scanners, returning zero to aggregator.")

        return response

    except Exception as _exception:
        info(f'Could not query endpoint {endpoint}, response {_exception}')