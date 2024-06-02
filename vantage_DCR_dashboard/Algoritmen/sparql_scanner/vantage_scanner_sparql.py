import time
import requests

import pandas as pd

from vantage6.tools.util import info
from io import StringIO


def master(client, data, organization_ids=None):
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
            'method': 'scanner_sparql_partial',
            'kwargs': {
            }
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

    # Combine the partials to a global count
    global_counts = {}

    for partial_results in results:
        # iterate through the unique values e.g., female and male sex
        for unique_values in partial_results.keys():
            # if unique value is not yet present add it
            if unique_values not in global_counts:
                global_counts[unique_values] = 0
            # sum the counts to existing counts
            global_counts[unique_values] += partial_results[unique_values]

    return global_counts


def RPC_scanner_sparql_partial(data):
    """
    Compute the average partial
    The data argument is supposed to contain a txt file with the ip address of the SPARQL-endpoint.

    :param data: define the SPARQL-endpoint in a csv file under column 'endpoint'; will read first row
    :param str predicate: name of the column to take the average of
    :return: local sum and local count
    """
    # extract the predicate from the dataframe.
    info(f'Extracting scanners')

    # extract the endpoint from the dataframe; vantage6 3.7.3  assumes csv as input and reads it as dataframe
    if len(data['endpoint']) != 1:
        info('Multiple endpoints defined in data file, using first occurrence')
    endpoint = str(data['endpoint'].iloc[0])

    # extract data and calculate at the sparql endpoint to ensure no data crosses over
    response = extract_scanners_via_sparql(endpoint)

    # transcribe the row counts
    info('Retrieving scanner counts')

    scanner_values = response["scanner"].value_counts(dropna=False).keys().tolist()
    scanner_counts = response["scanner"].value_counts(dropna=False).tolist()
    result = dict(zip(scanner_values, scanner_counts))

    # return the values as a dictionary
    return result

def extract_scanners_via_sparql(endpoint):
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
    query = f"""
        PREFIX lidicom: <https://johanvansoest.nl/ontologies/LinkedDicom/>
           SELECT ?scanner WHERE {{ 
               ?series lidicom:T00080070 ?scanner. 
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

