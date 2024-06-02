import pandas as pd
import hashlib


def convert_count_dict_to_dataframe(data_dict, filters, organisation_ids, existing_df=None):
    """
    Convert a dictionary of data into a Pandas DataFrame.

    :param dict data_dict: a dictionary where keys are category names, and values are dictionaries of categories and values.
    :param dict filters: a dictionary of the filters that were applied on the specific query
    :param list organisation_ids: a list of organisations that the task was run on, this list is converted to a hash id.
    This is necessary to create a unique id for the specific combination of organisations that the result belongs to
    :param pandas.DataFrame existing_df: An existing DataFrame to which the generated DataFrame will be appended.
            Defaults to None.

    :return pd.DataFrame: A DataFrame containing the converted data.

    Example:
        existing_dataframe = pd.DataFrame()  # Initialise an existing DataFrame (or provide an existing one)
        sample_data = {
            "A_count": {
                "0.0": 345,
                "1.0": 575
            },
            "B_count": {
                "0.0": 367,
                "1.0": 373
            },
        }

        result_dataframe = convert_dict_to_dataframe(sample_data, existing_dataframe)
        print(result_dataframe)
    """
    # initialise an empty list to store DataFrames
    dfs = []

    # iterate through the dictionaries and convert them to DataFrames
    for key, values_dict in data_dict.items():
        df = pd.DataFrame({
            "Categories": list(values_dict.keys()),
            "Values": list(values_dict.values())
        })

        # create a hash of the organisation_ids and the category name
        df["HashIdentifier"] = hash_information(key[:key.rfind('_count')], filters, organisation_ids)
        dfs.append(df)

    # concatenate all DataFrames into one
    final_df = pd.concat(dfs, ignore_index=True)

    # if an existing DataFrame is provided, append the generated DataFrame to it
    if isinstance(existing_df, pd.DataFrame):
        final_df = pd.concat([existing_df, final_df], ignore_index=True)

    return final_df


def convert_scan_dict_to_dataframe(data_dict, organisation_ids, existing_df=None):
    """
    Convert a dictionary of data into a Pandas DataFrame.

    :param dict data_dict: a dictionary where keys are category names, and values are dictionaries of categories and values.
    :param dict filters: a dictionary of the filters that were applied on the specific query
    :param list organisation_ids: a list of organisations that the task was run on, this list is converted to a hash id.
    This is necessary to create a unique id for the specific combination of organisations that the result belongs to
    :param pandas.DataFrame existing_df: An existing DataFrame to which the generated DataFrame will be appended.
            Defaults to None.

    :return pd.DataFrame: A DataFrame containing the converted data.

    Example:
        existing_dataframe = pd.DataFrame()  # Initialise an existing DataFrame (or provide an existing one)
        sample_data = {"A": 2, "B": 4}
        result_dataframe = convert_dict_to_dataframe(sample_data, existing_dataframe)
        print(result_dataframe)
    """

    # iterate through the dictionaries and convert them to DataFrames
    df = pd.DataFrame({
        "Scanners": list(data_dict.keys()),
        "Counts": list(data_dict.values())
    })

    # create a hash of the organisation_ids and the category name
    df["HashIdentifier"] = hash_information(organisation_ids)

    # if an existing DataFrame is provided, append the generated DataFrame to it
    if isinstance(existing_df, pd.DataFrame):
        df = pd.concat([existing_df, df], ignore_index=True)

    return df


def hash_information(*information_to_hash):
    """
    Turn the information of a query into a sha256 hash, such as variable name, filters, and organisations
    Supports multiple types including integers, floats, strings, lists, and dictionaries

    :param any information_to_hash: information that is to be put in the hash
    :return: sha256 hash as string of all the information that was provided to the function
    """
    # create an empty string to add information to
    hash_info = ""

    # cycle through the information that is to be hashed
    for info in information_to_hash:
        if isinstance(info, (int, float)):
            info = str(info)
        # concatenate the information as strings
        hash_info += str(info)

    # generate a single hash for all information
    hash_identifier = hashlib.sha256(hash_info.encode()).hexdigest()
    return hash_identifier


def convert_heatmap_to_appropriate_dataframe(output_dataframe, organisation_ids, roi_names, existing_df=None):
    """

    :param roi_names:
    :param dict output_dataframe: A dictionary where keys are category names, and values are dictionaries of categories and values.
    :param list organisation_ids: a list of organisations that the task was run on, this list is converted to a hash id.
    This is necessary to create a unique id for the specific combination of organisations that the result belongs to
    :param pandas.DataFrame existing_df: An existing DataFrame to which the generated DataFrame will be appended.
            Defaults to None.

    :return pd.DataFrame: A DataFrame containing the converted data.
    """
    # create a hash of the organisation_ids
    organisation_hash = hashlib.sha256((str(tuple(organisation_ids)) + roi_names).encode()).hexdigest()

    # add extra columns for ROI filter and the organisation hash
    output_dataframe["ROI"] = roi_names
    output_dataframe["OrganisationHash"] = organisation_hash

    placeholder_heatmap_dict = {organisation_hash: {}}
    placeholder_heatmap_dict[organisation_hash]['columns'] = list(output_dataframe.columns)
    placeholder_heatmap_dict[organisation_hash]['ROI'] = roi_names

    # add filter or something alike

    # if an existing DataFrame is provided, append the generated DataFrame to it
    if isinstance(existing_df, pd.DataFrame):
        output_dataframe = pd.concat([existing_df, output_dataframe], ignore_index=True)

    return output_dataframe, placeholder_heatmap_dict
