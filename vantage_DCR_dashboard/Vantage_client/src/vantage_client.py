import json
import os
import sys
import subprocess
import time

from vantage6.client import Client

# private module
import config as config


class Vantage6Client:
    def __init__(self):
        """"""
        self.Client = None
        self.Tasks = {}
        self.Results = {}
        self.Dashboard = None

        self.Directory = os.getcwd()

        # ensure path is present
        if os.path.exists(os.path.join(self.Directory, '../output')) is False:
            os.mkdir(os.path.join(self.Directory, '../output'))
        self.OutputPath = os.path.join(self.Directory, '../output')

    def login(self, username=None, password=None):
        """
        login to the specified Vantage6 server

        :param str username: Vantage6 Username
        :param str password: Vantage6 Password
        :return:
        """
        # Initialize the client object, and run the authentication
        self.Client = Client(config.server_url, config.server_port, config.server_api, verbose=True)

        # retrieve login details
        if isinstance(username, str) is False:
            if isinstance(config.username, str) is False:
                username = input(f'Please provide your username for server\n{config.server_url}')
            else:
                username = config.username

        if isinstance(password, str) is False:
            if isinstance(config.password, str) is False:
                password = input(f'Please provide the password for user: {username}')
            else:
                password = config.password

        self.Client.authenticate(username, password)

        # Optional: set up the encryption, if you have an organization_key
        self.Client.setup_encryption(config.organization_key)

    def varsha_benedetta(self, column_names=None, name=None, description=None, check_results=True, save_results=True):
        """"""

        if name is None:
            name = "aaa"

        if description is None:
            description = "This is an example algorithm call"

        input_ = {
            'master': 1,
            'method': 'master',
            'kwargs': {
                'expl_var': ['original_shape_MeshVolume', 'original_firstorder_Energy', 'original_shape_Sphericity']
            },
        }

        task = self.Client.task.create(name="testing",
                                       description="test connection",
                                       image="coxphl1/vtg_corr:latest",
                                       collaboration=1,
                                       input=input_,
                                       organizations=[2, 3])

        if check_results:
            filename = None
            if save_results:
                filename = f'{name}_{column_names}.json'
            self.retrieve_results(task, name, filename)

        self.Tasks.update({name: task})

    def take_summary(self, column_names=None, name=None, description=None, check_results=True, save_results=True):
        """"""
        if column_names is None:
            # The `columns` in the datasets you want to summarize and specify if
            # they are categorical ("category" or "c") or numeric ("numeric" or
            # "n")
            column_names = {"pf": "n", "rf": "n", "ef": "n", "cf": "n", "sf": "n", "ql": "n"}

        if name is None:
            name = "Summary algorithm"

        if description is None:
            description = "This is an example algorithm call"

        input_ = {
            "master": True,
            "method": "master",

            # kwargs which are inserted into the algorithm
            "kwargs": {

                "columns": column_names,
                # Optionally, the organizations you want to include. They need to be
                # within the collaboration. By default all organizations in the
                # collaboration are included.
                # "organizations_to_include": [1, 2, 3],  # default: 'ALL'

                # Optionally, the subset you want to know the summary about. E.g. you only
                # want to include patients with a certain diagnose.
                # "subset": {"diagnose": "cancer"}
            }
        }

        # Send the task to the central server
        task = self.Client.task.create(name=name,
                                       description=description,
                                       collaboration=1,
                                       organizations=[2],
                                       image="harbor.vantage6.ai/algorithms/summary",
                                       input=input_,
                                       organization_ids=[2])
        if check_results:
            filename = None
            if save_results:
                filename = f'{name}_{column_names}.json'
            self.retrieve_results(task, name, filename)

        self.Tasks.update({name: task})

    def take_average(self, column_name=None, collaboration=None, aggregating_organisation=None,
                     name=None, description=None, check_results=True, save_results=True):

        """
        Take the average of a given column

        :param string column_name: name of column to take the average off
        :param integer collaboration: collaboration to run the task in
        :param list aggregating_organisation: organisation(s) to aggregate the task in
        :param string name: define the name of the task
        :param string description: provide a description of the task
        :param boolean check_results: specify whether to check for results
        :param boolean save_results:  specify whether to save the results as JSON file
        """
        if collaboration is None:
            collaboration = 1

        if aggregating_organisation is None:
            aggregating_organisation = [3]

        if column_name is None:
            column_name = 'pf'

        if name is None:
            name = 'Average Physical functioning'

        if description is None:
            description = 'Retrieve the average Physical functioning score'

        input_average = {'method': 'master',
                         'kwargs': {'column_name': column_name},
                         'master': True}

        task = self.Client.task.create(name=name,
                                       description=description,
                                       collaboration=collaboration,
                                       organizations=aggregating_organisation,
                                       image="harbor2.vantage6.ai/demo/average",
                                       input=input_average,
                                       data_format='json')

        if check_results:
            filename = None
            if save_results:
                filename = f'{name}_{column_name}.json'
            self.retrieve_results(task, name, filename)

        self.Tasks.update({name: task})

    def take_average_predicate_sparql(self, predicates=None, filters=None, collaboration=None, organisation_ids=[4],
                                      aggregating_organisation=None, name=None, description=None,
                                      check_results=True, save_results=True):
        """
        Take the average of a given predicate using SPARQL

        :param list predicates: predicates to query and take the average off
        :param
        :param integer collaboration: collaboration to run the task in
        :param list organisation_ids: organisations to run the task in
        :param list aggregating_organisation: organisation(s) to aggregate the task in
        :param string name: define the name of the task
        :param string description: provide a description of the task
        :param boolean check_results: specify whether to check for results
        :param boolean save_results:  specify whether to save the results as JSON file
        """
        if collaboration is None:
            collaboration = 2

        if aggregating_organisation is None:
            aggregating_organisation = [3]

        if predicates is None:
            predicates = ['roo:hasphysicalfunctioning']

        if name is None:
            name = 'Average Physical functioning - SPARQL'

        if description is None:
            description = 'Retrieve the average Physical functioning of the ' \
                          'Physical Functioning predicate through SPARQL'

        input_average = {'method': 'master',
                         'kwargs': {'predicates': predicates,
                                    'filters': filters,
                                    'organization_ids': organisation_ids},
                         'master': True}

        task = self.Client.task.create(name=name,
                                       description=description,
                                       collaboration=collaboration,
                                       organizations=aggregating_organisation,
                                       image="jhogenboom/average_sparql",
                                       input=input_average,
                                       data_format='json',
                                       database='rdf')

        if check_results:
            filename = None
            if save_results:
                filename = f'{name}_{predicates}.json'
            self.retrieve_results(task, name, filename)

        self.Tasks.update({name: task})

    def perform_generalised_linear_regression(self, formula=None, categorical_variables=None,
                                              family=None, tolerance=None, max_iterations=None,
                                              collaboration=None, organisation_ids=[2, 3, 5, 6],
                                              aggregating_organisation=None, name=None, description=None,
                                              check_results=True, save_results=True):
        """

        :param formula: 'outcome ~ explanatory_variable_1 + explanatory_variable_2 + et cetera'
        :param categorical_variables: 'explanatory_variable_2': {'type': 'factor',
                                                   'levels': ['General', 'Vocational', 'Academic']}
        :param collaboration:
        :param organisation_ids:
        :param aggregating_organisation:
        :param name:
        :param description:
        :param check_results:
        :param save_results:
        :return:
        """
        if isinstance(formula, str) is False:
            formula = "BIneg_new ~ alg_v1b + " \
                      "partner + " \
                      "pf + " \
                      "rf + " \
                      "ef + " \
                      "cf + " \
                      "sf + " \
                      "ql"

        if isinstance(categorical_variables, dict) is False:
            categorical_variables = None
            # {
            # excluded variables type as in the smaller synthetic sets categories might lack and throw errors

            # 'edu': {
            #     'type': 'factor',
            #     'levels': ['3', '2', '1']},
            # 'tumsoort_hoofd_new2': {
            #     'type': 'factor',
            #     'levels': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']},
            # 'stadium_opgeschoond': {
            #     'type': 'factor',
            #     'levels': ['1', '2', '3', '4', '99']},
            # 'tijddiagngroep': {
            #     'type': 'factor',
            #     'levels': ['1', '2', '3']},
            # 'BMIgroups2': {
            #     'type': 'factor',
            #     'levels': ['1', '2', '3', '4']}
            #     }

            if isinstance(family, str) is False:
                family = 'binomial'

            if isinstance(tolerance, (int, float)) is False:
                tolerance = 1e-08

            if isinstance(max_iterations, int) is False:
                max_iterations = 25

            if isinstance(collaboration, int) is False:
                collaboration = 2

            if isinstance(aggregating_organisation, list) is False:
                aggregating_organisation = [8]

            if isinstance(name, str) is False:
                name = 'Logistic Regression'

            if isinstance(description, str) is False:
                description = f'Logistic regression for formula:\n {formula}'

            # this step is done seperately whereas None might cause errors
            if isinstance(organisation_ids, list):
                input_regression = {'master': True,
                                    'method': 'dglm',
                                    'args': [],
                                    'kwargs': {
                                        'formula': formula,
                                        'types': categorical_variables,
                                        'family': family,
                                        'tol': tolerance,
                                        'maxit': max_iterations,
                                        'organizations_to_include': organisation_ids},
                                    'output_format': 'json'}
            else:
                input_regression = {'master': True,
                                    'method': 'dglm',
                                    'args': [],
                                    'kwargs': {
                                        'formula': formula,
                                        'types': categorical_variables,
                                        'family': family,
                                        'tol': tolerance,
                                        'maxit': max_iterations},
                                    'output_format': 'json'}

            # Sending the analysis task to the server
            task = self.Client.task.create(collaboration=collaboration,
                                           organizations=aggregating_organisation,
                                           name=name,
                                           description=description,
                                           image='jhogenboom/glm_csv:1.1.0',
                                           input=input_regression,
                                           data_format='json')

            if check_results:
                filename = None
            if save_results:
                filename = f'{name}_{formula}.json'
            self.retrieve_results(task, name, filename)

            self.Tasks.update({name: task})

    def compute_dashboard(self, columns_to_count=None, columns_to_describe=None, column_to_stratify=None,
                          organisation_ids=None, name=None, description=None,
                          check_results=True, save_results=True):
        """"""
        # placeholder for column name
        column_name = ''

        if name is None:
            name = 'Dashboard algorithm'

        if description is None:
            description = 'Retrieve values to place in the dashboard'

        input = {'method': 'master',
                 'master': True,
                 'kwargs': {'columns_to_count': columns_to_count,
                            'columns_to_describe': columns_to_describe,
                            'column_to_stratify': column_to_stratify,
                            'organization_ids': organisation_ids}}

        task = self.Client.task.create(name=name,
                                       description=description,
                                       collaboration=1,
                                       organizations=[6],
                                       image="varshagouthamchand/v6_dash_master",
                                       input=input,
                                       data_format='json',
                                       database='default')

        if check_results:
            filename = None
            if save_results:
                filename = f'{name}_{column_name}.json'
            filepath = self.retrieve_results(task, name, filename, True)

        self.Tasks.update({name: task})

    def compute_count_sparql(self, predicates=None, filters=None, collaboration=None, organisation_ids=None,
                             name=None, description=None, check_results=True, save_results=True):
        """"""
        if predicates is None:
            predicates = ['roo:P100018']

        if isinstance(collaboration, int) is False:
            collaboration = 1

        if name is None:
            name = f'Counts of {predicates} - SPARQL'

        if description is None:
            description = 'Retrieve the unique values and their counts using a SPARQL query'

        input_counts_sparql = {'method': 'master',
                               'master': True,
                               'kwargs': {'predicates': predicates,
                                          'organization_ids': organisation_ids,
                                          'filters': filters}}

        # Sending the analysis task to the server
        task = self.Client.task.create(collaboration=collaboration,
                                       organizations=[2],
                                       name=name,
                                       description=description,
                                       image='varshagouthamchand/count_pie_sparql:latest',
                                       input=input_counts_sparql,
                                       data_format='json',
                                       database='rdf')

        if check_results:
            filename = None
            if save_results:
                filename = f'{name}_{predicates}.json'
            self.retrieve_results(task, name, filename)

        self.Tasks.update({name: task})

    def compute_scanner_sparql(self, collaboration=None, organisation_ids=None, name=None,
                               description=None, check_results=True, save_results=True):
        """"""

        if isinstance(collaboration, int) is False:
            collaboration = 1

        if name is None:
            name = f'Counts of scanners - SPARQL'

        if description is None:
            description = 'Retrieve the scanners and their counts using a SPARQL query'

        input_counts_sparql = {'method': 'master',
                               'master': True,
                               'kwargs': {'organization_ids': organisation_ids}}

        # Sending the analysis task to the server
        task = self.Client.task.create(collaboration=collaboration,
                                       organizations=[2],
                                       name=name,
                                       description=description,
                                       image='varshagouthamchand/count_scanners_sparql:latest',
                                       input=input_counts_sparql,
                                       data_format='json',
                                       database='rdf')

        if check_results:
            filename = None
            if save_results:
                filename = f'scan.json'
            self.retrieve_results(task, name, filename)

        self.Tasks.update({name: task})

    def compute_features_sparql(self, feature, collaboration=None, organisation_ids=None, name=None,
                                description=None, check_results=True, save_results=True):
        """"""

        if isinstance(collaboration, int) is False:
            collaboration = 1

        if name is None:
            name = f'Radiomics feature values - SPARQL'

        if description is None:
            description = 'Retrieve the features and their values using a SPARQL query'

        input_features_sparql = {'method': 'master',
                                 'master': True,
                                 'kwargs': {'feature': feature,
                                            'organization_ids': organisation_ids}}

        # Sending the analysis task to the server
        task = self.Client.task.create(collaboration=collaboration,
                                       organizations=[2],
                                       name=name,
                                       description=description,
                                       image='varshagouthamchand/sparql_rad_features:latest',
                                       input=input_features_sparql,
                                       data_format='json',
                                       database='rdf')

        if check_results:
            filename = None
            if save_results:
                filename = f'scan.json'
            self.retrieve_results(task, name, filename)

        self.Tasks.update({name: task})

    def compute_hm_sparql(self, expl_vars, censor_col, roitype, organisation_ids=None, collaboration=None,
                          description=None, name=None, check_results=True, save_results=True):
        """"""

        if isinstance(collaboration, int) is False:
            collaboration = 1

        if name is None:
            name = f'Heatmap - SPARQL'

        if description is None:
            description = 'Retrieve the correlation matrix using a SPARQL query'

        input_hm_sparql = {'method': 'master',
                           'master': True,
                           'kwargs': {'expl_vars': expl_vars,
                                      'censor_col': censor_col,
                                      'roitype': roitype,
                                      'organization_ids': organisation_ids}}

        # Sending the analysis task to the server
        task = self.Client.task.create(collaboration=collaboration,
                                       organizations=[2],
                                       name=name,
                                       description=description,
                                       image='varshagouthamchand/v6_hm',
                                       input=input_hm_sparql,
                                       data_format='json',
                                       database='rdf')

        if check_results:
            filename = None
            if save_results:
                filename = f'hm.json'
            self.retrieve_results(task, name, filename)

        self.Tasks.update({name: task})

    def retrieve_results(self, task, name, filename, return_filepath=False):
        """"""
        print("Waiting for results")
        task_id = task['id']
        task_info = self.Client.task.get(task_id)
        while not task_info.get("complete"):
            task_info = self.Client.task.get(task_id, include_results=True)
            print("Waiting for results")
            time.sleep(3)

        print("Results are ready!")

        result_id = task_info['id']
        result_info = self.Client.result.list(task=result_id)

        output_data = result_info['data'][0]['result']

        #print(f'\n################################\nResult of query: {output_data}\n################################\n')

        self.Results.update({name: output_data})

        if isinstance(filename, str):
            filepath = f'{self.OutputPath}{os.path.sep}{filename}'
            with open(filepath, 'w') as file_path:
                json.dump(output_data, file_path, indent=4)

            if return_filepath:
                return filepath
