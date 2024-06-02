import requests
import time

import pandas as pd

from vantage6.tools.util import info
from io import StringIO

def master(client, data, predicates, filters=None, organization_ids=None):
    """
    Combine partials to global model for the provided predicate and specified organisations

    :param client: vantage client
    :param data: data defined in node config file
    :param list predicates: predicates to retrieve the counts for, defined by user
    :param dict filters:
    :param list organization_ids: organisations to run the algorithm on, defined by user
    :return: global count of unique values
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

    # compose SPARQL query
    info('Composing SPARQL query')

    # check that predicates and filters are lists of strings
    if isinstance(predicates, list) is False:
        # force it into a list if not already a list
        predicates = [predicates]

        if isinstance(predicates[0], str) is False:
            info(f'Unsupported format provided for predicate (input was {predicates}) use a list with strings instead')

    if isinstance(filters, dict) is False:
        if filters is not None:
            info(f'Unsupported format provided for filters (input was {filters}), check docstring for proper formatting')
        # force it into a list if not already a list
        filters = {}

    queries = compose_sparql_query(predicates, filters)

    # Request all (specified) organisations parties to compute their partial
    info('Requesting partial counts using sparql')
    task = client.create_new_task(
        input_={
            'method': 'count_sparql_partial',
            'kwargs': {
                'queries': queries
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

    definitive_result = {}
    global_counts = {}

    # iterate through the partial results
    for predicate in predicates:
        for partial_results in results:
            # iterate through the unique values e.g., female and male sex
            for unique_values in partial_results[predicate]:
                # if unique value is not yet present add it
                if unique_values not in global_counts:
                    global_counts[unique_values] = 0
                # sum the counts to existing counts
                global_counts[unique_values] += partial_results[predicate][unique_values]

        # wrap global counts in a dict with the appropriate variable key
        definitive_result.update({f'{predicate}_count': global_counts})

    return definitive_result


def RPC_count_sparql_partial(data, queries):
    """
    Compute the partial counts
    The data argument is supposed to contain a txt file with the ip address of the SPARQL-endpoint.

    :param data: define the SPARQL-endpoint in a csv file under column 'endpoint'; will read first row
    :param dict queries: the SPARQL-query that is to be sent to the graphdb endpoint
    :return: local unique values and local count
    """
    # extract the predicate from the dataframe.
    info(f'Extracting column counts using a SPARQL-query')

    # extract the endpoint from the dataframe; vantage6 3.7.3  assumes csv as input and reads it as dataframe
    if len(data['endpoint']) != 1:
        info('Multiple endpoints defined in data file, using first occurrence')
    endpoint = str(data['endpoint'].iloc[0])

    # extract data and calculate at the sparql endpoint to ensure no data crosses over
    responses = extract_count_via_sparql(queries, endpoint)

    # transcribe the sum, and row counts
    info('Retrieving partial unique values and their counts')

    # return the values as a dictionary
    return responses


def compose_sparql_query(predicates_to_query, filters_to_apply):
    """
    Compose a sparql query that takes the sum and count of a list of predicates whilst including specified filters

    :param list predicate_to_query: provide the predicates to take the count of
    for example as: "roo:123"
    :param dict filters_to_apply: provide a dict of the predicate to filter on, including its type and ncit value to filter
    for example as:
    {
    "roo:001": ['C100', 'C200']
    }
    :return: a list of queries for every predicate including every desired filter
    """
    predicates_with_neoplasm = ['roo:P100244', 'roo:P100242', 'roo:P100241', 'roo:P100219', 'roo:P100202']

    # define prefixes as a string
    prefixes = """ 
            PREFIX dbo: <http://um-cds/ontologies/databaseontology/>
            PREFIX roo: <http://www.cancerdata.org/roo/>
            PREFIX ncit: <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"""

    # initialise an empty list to store individual SPARQL queries
    queries = {}

    # loop through the input list and filter dictionary to generate SPARQL queries
    for predicate in predicates_to_query:
        # initialise an empty query string for this input
        query = f"{prefixes}\n"
        query += f"""
                    SELECT ?unique_values (count (DISTINCT(?patientID)) as ?count)
                    WHERE {{
                        ?patient roo:P100061 ?Subject_label.
                        ?Subject_label dbo:has_cell ?Subject_cell.
                        ?Subject_cell dbo:has_value ?patientID.
                        ?patient dbo:has_clinical_features ?clinical.
                        ?clinical dbo:has_table ?clin_table.
                """
        if any(x in predicates_to_query for x in predicates_with_neoplasm):
            query += f"""
                        ?clin_table roo:P100029 ?neoplasm.
                        ?neoplasm """ + predicate + """ ?variable.
                        ?variable dbo:has_cell ?variable_cell.
                        ?variable_cell a ?type.
                        """
        else:
            query += f"""
                        ?clin_table """ + predicate + """ ?variable.
                        ?variable dbo:has_cell ?variable_cell.
                        ?variable_cell a ?type.
                        """
        # filter specific pattern
        for key, value_list in filters_to_apply.items():
            filters = ''
            values = '|'.join([f'http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#{value}' for value in value_list])
            filters = f'FILTER regex(str(?type), ("{values}"))'

            query += f"""
                {filters}
                BIND(strafter(str(?type), "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#") AS ?unique_values)
            """

        query += "} GROUP BY (?unique_values)\n"  # closing brace for the initial pattern

        # update the dictionary with the predicate and the associated query
        queries.update({predicate: query})

    return queries

def extract_count_via_sparql(queries, endpoint):
    """
    Extracts data via SPARQL of given predicate at specified endpoint

    :param dict queries: the SPARQL-queries that are to be sent to the graphdb endpoint
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

    responses = {}
    for predicate, query in queries.items():
        # attempt to query
        try:
            query_response = requests.post(endpoint, data=f'query={query}',
                                           headers={'Content-Type': 'application/x-www-form-urlencoded'})

            # check whether there is any data with the specified predicate
            response = pd.read_csv(StringIO(query_response.text))

            # check validity of response
            if response.empty:
                info(
                    f"Endpoint: {endpoint} does not contain data for predicate: {predicate}, returning zero to aggregator.")

            # add the response for the specific predicate
            responses.update({predicate: dict(response.to_numpy())})

        except Exception as _exception:
            info(f'Could not query endpoint {endpoint}, response {_exception}')

    return responses