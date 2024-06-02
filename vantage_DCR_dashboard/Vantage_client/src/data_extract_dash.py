"""
Data extraction methods for dashboard
V. Gouthamchand and J. Hogenboom - March 2023
"""
import json
import pandas as pd

# private module
import Vantage_client.src.vantage_client as client_vantage


def read_output(path_to_output):
    """
    Read a json or csv file

    :param str path_to_output: Path to output file that is to be read
    :return: contents of JSON file or pandas.DataFrame for 'csv' files
    """
    if '.json' in path_to_output:
        # read JSON file
        with open(f'{path_to_output}', 'r') as filepath:
            output_data = json.load(filepath)
        return output_data
    elif '.csv' in path_to_output:
        # TODO add CSV support via pandas
        pass
    else:
        return None


def retrieve_existing_output_data(output_dictionary, variable_to_stratify, string_identifier,
                                  outcomes_stratified_variable, variable_to_extract):
    """
    Retrieve data per variable on a certain level of stratification
    e.g., retrieve all the counts for having a certain outcome

    :param dictionary output_dictionary: dictionary containing the data to select
    :param str variable_to_stratify: define the variable that was stratified on
    :param str string_identifier: define the string that can be used to identify those keys with stratified data
    :param str outcomes_stratified_variable: define the outcomes to fetch,
    e.g., (negative body image yes no, tumour type)
    :param str variable_to_extract: define what data to extract
    e.g., those with therapy or not (1.0 or 0) or the mean
    :return: pandas.DataFrame containing the desired output data
    """
    if isinstance(output_dictionary, str):
        output_dictionary = read_output(output_dictionary)

    # retrieve the variables to plot
    variables = {variable[:variable.rfind('_')]: {}
                 for variable in output_dictionary.keys()
                 if f'{string_identifier}{variable_to_stratify}' in variable}

    # retrieve the data to extract for a certain therapy outcome
    for variable in variables.keys():
        variables[variable].update({outcome: output_dictionary[f'{variable}_{outcome}'][f'{variable_to_extract}']
                                    for outcome in outcomes_stratified_variable})

    # convert to pandas
    variable_data_frame = pd.DataFrame.from_dict(variables, orient='index')

    return variable_data_frame


def retrieve_non_existing_output_data(existing_data, column_to_count):
    """
    Retrieve the counts of a certain variable by sending a task via your vantage6 configured client
    TODO increase support of data to extract

    :param pandas.DataFrame existing_data: dataframe to which the new data will be concatenated
    :param str column_to_count: variable name which to count
    :return: pandas.DataFrame containing the existing plus new information
    """
    task_name = f'Dashboard request of counts for {column_to_count}'

    # initialise a client
    user = client_vantage.Vantage6Client()
    user.login()

    # run the task and extract the output
    user.compute_dashboard(name=task_name, columns_to_count=[column_to_count], save_results=False)
    results = user.Results[task_name]

    # TODO avoid reshape
    df = pd.DataFrame.from_dict(results, orient='index')
    df['therapy'] = df.index

    df_new = pd.concat([existing_data, df], axis=0)
    return df_new
