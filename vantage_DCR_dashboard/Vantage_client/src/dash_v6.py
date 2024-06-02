import dash
import hashlib
import vantage6.client

import numpy as np
import plotly.express as px
import pandas as pd

from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
import scipy.cluster.hierarchy as sch

# private module
import miscellaneous
import vantage_client

# Set a global variable for the font
GLOBAL_FONT = "Times New Roman, sans-serif"


class Dashboard:
    def __init__(self):
        """"""
        # settings
        self.ColourSchemeContinuous = px.colors.sequential.Agsunset
        self.ColourSchemeCategorical = px.colors.sequential.Agsunset

        # vantage components
        self.Vantage6User = vantage_client.Vantage6Client()
        self.Organisations = None
        # organisation names are ideally retrieved from the client, but some standard names will have to be in place
        # currently names have to be changed to match specific node names
        self.OrganisationsNames = ['HN1_Maastro', 'Montreal', 'Toronto', 'HN3_Maastro', 'HCG', 'TMH']
        self.Organisations_ids_to_query = []

        self.roi_names = {'GTV Primary': 'C100346',
                          'GTV Node': 'C100347'}

        self.codedict = {
            "C00000": "Unknown", "C48737": "Tx", "C48719": "T0", "C48720": "T1", "C48724": "T2", "C20197": "Male",
            "C16576": "Female", "C48728": "T3", "C48732": "T4", "C48705": "N0", "C48706": "N1", "C48786": "N2", "C48714": "N3",
            "C28554": "Dead", "C37987": "Censored", "C128839": "HPV Positive", "C100346": "Primary", "C100347": "Node",
            "C12762": "Oropharynx", "C12420": "Larynx", "C12246": "Hypopharynx", "C12423": "Nasopharynx", "C12421": "Oral cavity",
            "C150211": "Larynx", "C4044": "Larynx", "C20000": 1, "C30000": 0, "C40000": 1, "C50000": 0}

        self.filter_dict = {'roo:P100018': ['C16576', 'C20197'],
                            'roo:P100244': ['C48719', 'C48720', 'C48724', 'C48728', 'C48732'],
                            'roo:P100242': ['C48705', 'C48706', 'C48786', 'C48714'],
                            'roo:P100241': ['C48699', 'C48700'],
                            'roo:P100028': ['C28554', 'C37987'],
                            'roo:P100022': ['C128839', 'C131488'],
                            'roo:P100219': ['C27966', 'C28054', 'C27970', 'C27971'],
                            'roo:P100202': ['C12762', 'C12246', 'C12420', 'C12423', 'C00000'],
                            'roo:P100231': ['C94626', 'C15313']}

        self.Filters_to_apply = {}

        # this 'dataset' is used to display some data when the user has not been authenticated yet
        self._PlaceholderData = {"Not an actual variable_count": {"0.0": 2, "1.0": 4}}
        self.PlaceholderDataframe = miscellaneous.convert_count_dict_to_dataframe(self._PlaceholderData,
                                                                                  self.Filters_to_apply,
                                                                                  self.Organisations_ids_to_query)

        self._PlaceholderDataScanners = {"A": 2, "B": 4}
        self.PlaceholderDataframeScanners = miscellaneous.convert_scan_dict_to_dataframe(self._PlaceholderDataScanners,
                                                                                         self.Organisations_ids_to_query)

        self._PlaceholderDataHeatMap = {}
        self.PlaceholderDataFrameHeatMap, temp_dict = miscellaneous.convert_heatmap_to_appropriate_dataframe(
            pd.DataFrame(np.random.rand(10, 10), columns=[f'Column_{i}' for i in range(10)]),
            self.Organisations_ids_to_query, tuple(self.roi_names.values())[0])

        self._PlaceholderDataHeatMap.update(temp_dict)

        # Define the number of box plots and the number of data points for each boxplot (dummy)
        num_boxplot = 6
        num_data_points = 50

        self._PlaceholderDataFeatures = {f'Boxplot {i}': np.random.normal(loc=i, scale=1, size=num_data_points).tolist()
                                         for i in range(num_boxplot)}
        # content components
        self.DashboardTitle = ''
        self.DashboardTileTexts = ["3 countries", "5 institutions", "1814 patients"]

        self.pie_dropdown_variables = {'roo:P100018': 'Gender',
                                       'roo:P100244': 'T-stage',
                                       'roo:P100242': 'N-stage',
                                       'roo:P100241': 'M-stage',
                                       'roo:P100028': 'Survival Status',
                                       'roo:P100022': 'HPV Status',
                                       'roo:P100219': 'AJCC Stage',
                                       'roo:P100202': 'Tumour Location',
                                       'roo:P100231': 'Therapy given'}

        self.rad_options = self.read_rad_features()

        # refers to <folder_with_this_file>/assets/dashboard_aesthetics.css
        self.App = dash.Dash(__name__, external_stylesheets=['dashboard_aesthetics.css'])
        self.Layout = self.define_layout()
        self.register_callbacks()

    def define_layout(self):
        """"""
        self.App.layout = html.Div([
            dcc.Store(id='authentication-status', data=False),  # Store for login status

            html.Header([
                html.Div(className='primary-header', children=[
                    html.Div(id='input-container', className='input-container', children=[
                        dcc.Input(id='input-username', className='input-field', placeholder='Username', type='text'),
                        dcc.Input(id='input-password', className='input-field', placeholder='Password',
                                  type='password'),
                        html.Button('Submit', id='login-button', n_clicks=0, className='login-button')
                    ]),
                    # Welcome message
                    dcc.Markdown(id='welcome-message', className='welcome-message white-text', children=[],
                                 style={'font-family': GLOBAL_FONT}),
                ]),
            ]),

            # html.Div([
            # Text tiles
            #     html.Div(id='dashboard', className='dashboard', children=[
            #         html.H5(id='dashboard-title', className='dashboard-title', children=self.DashboardTitle),

            #        html.Div(id='tile-1', className='tile', children=[
            #            html.Div(id='tile-content-1', className='tile-content', children=self.DashboardTileTexts[0])
            #        ]),

            #        html.Div(id='tile-2', className='tile', children=[
            #            html.Div(id='tile-content-2', className='tile-content', children=self.DashboardTileTexts[1])
            #        ]),

            #        html.Div(id='tile-3', className='tile', children=[
            #            html.Div(id='tile-content-3', className='tile-content', children=self.DashboardTileTexts[2])
            #        ])
            #    ]),
            #]),

            # Centered container
            html.Div(className='graph-container', children=[
                html.Div(id='auth-lock', className='authentication-lock',
                         children='Please log in on the top left to explore the data'),
                html.Div([
                    html.Label("Select organization:",
                               style={'textAlign': 'center', 'fontSize': '23px', 'font-family': GLOBAL_FONT}),
                    dcc.Checklist(
                        id='institution-checklist',
                        options=[{'label': organisation, 'value': organisation} for organisation in
                                 self.OrganisationsNames],
                        value=[],
                        className='organisation-checklist',
                        inline=True
                    ),
                ], style={'textAlign': 'center', 'width': '100%', 'margin': '0 auto', 'marginTop': '20px',
                          'fontSize': '20px', 'font-family': GLOBAL_FONT}),

                html.Div(style={'display': 'flex'}, children=[
                    html.Div([
                        dcc.Dropdown(
                            id='dataset-variable',
                            value='Not an actual variable',
                            options=[{'label': variable_label, 'value': variable_identification}
                                     for variable_identification, variable_label in
                                     self.pie_dropdown_variables.items()],
                            clearable=False,
                            className='pie-drop-down'),
                        html.Div(id='tab-pie-content', className='graph-content'),
                    ], style={'display': 'flex', 'flexDirection': 'column', 'width': '50%'}),

                    # Bar chart
                    html.Div(id='tab-bar-content', className='graph-content', style={'width': '50%'}),
                ]),

                html.Br(),  # This creates a line break
                html.Div(style={'display': 'flex'}, children=[
                    html.Div([
                        dcc.Dropdown(
                            id='box-plot-dropdown',
                            options=[{'label': option, 'value': option} for option in self.rad_options],
                            # The options will be populated dynamically
                            value='Fmorph.sph.sphericity',  # The initial value of the dropdown
                            clearable=False,
                            searchable=True,  # Make the dropdown searchable
                            className='box-plot-drop-down',
                            style={'width': '50%'}),
                        html.Div(id='box-plot-content', className='graph-content'),
                    ], style={'display': 'flex', 'flexDirection': 'column', 'width': '100%'}),
                ]),

                html.Div(id='query-trigger-not-for-display'),

                # Display the heatmap below the tabs
                html.Br(),  # This creates a line break
                html.Div(className='graph-shadow', children=[
                    html.Div([
                    ], style={'flex': '1'}),
                    html.Div([
                        html.Br(),
                        html.Label("Select ROI for heatmap:", style={'fontSize': '18px', 'font-family': GLOBAL_FONT}),
                        dcc.RadioItems(
                            id='roi-checklist',
                            options=[{'label': roi_label, 'value': roi_value} for roi_label, roi_value in
                                     self.roi_names.items()],
                            value=tuple(self.roi_names.values())[0],  # Initially select first
                            style={'marginRight': '50px', 'marginTop': '10px', 'fontSize': '16px',
                                   'font-family': GLOBAL_FONT},  # Add some spacing between checkboxes,
                            inline=True  # Display the options horizontally
                        ),
                    ], style={'marginLeft': '15px','fontSize': '18px', 'font-family': GLOBAL_FONT,
                              'background-color': 'whitesmoke'}),
                    html.Div(id='heatmap-content', className='graph-content'),
                ], style={'background-color': 'whitesmoke'}),

            ])
        ])

    def read_rad_features(self):
        with open('rad_features', 'r') as file:
            features = file.readlines()
        # Remove any newline characters
        features = [feature.rstrip() for feature in features]
        return features

    def register_callbacks(self):
        """"""

        @self.App.callback(
            [Output('authentication-status', 'data'),
             Output('input-container', 'style'),
             Output('welcome-message', 'children'),
             Output('auth-lock', 'style')],
            [Input('login-button', 'n_clicks')],
            [State('input-username', 'value'),
             State('input-password', 'value')]
        )
        def authenticate(n_clicks, username, password):
            """"""
            if n_clicks > 0:
                try:
                    # log in
                    self.Vantage6User.login(username, password)
                    self.Organisations = self.Vantage6User.Client.organization.list()

                    # Successful authentication, return True (logged in) and empty style for input container
                    welcome_message = [f'Welcome {username}!']
                    return True, {'display': 'none'}, welcome_message, {'display': 'none'}
                except vantage6.client.AuthenticationException:
                    return False, {'display': 'block'}, [], {'display': 'block'}
            # Default: Display the login input fields, an empty welcome message, and hide the button
            return False, {'display': 'block'}, [], {'display': 'block'}

        @self.App.callback(
            Output('query-trigger-not-for-display', 'children'),
            [Input('authentication-status', 'data'),
             Input('institution-checklist', 'value')]
        )
        def select_organisations(authentication_status, organisation_to_include):
            """
            Transcribe the user's selection of organisations to ids that can be used in the Vantage6 Python client
            by directly accessing the Vantage6 Python client which are then stored it in a class object.

            The dcc.CheckList was not directly used as a callback to use Vantage6's Python client to ensure that:
             - dummy data can be displayed whilst not being an authenticated user
             - new organisations are queryable without hard-coding any of their ids.

            :param bool authentication_status: ensure that the user is authenticated to retrieve the organisation ids
            :param list organisation_to_include: list of organisations that the user has selected on the dashboard
            :return: an empty string, the query trigger is solely used to prioritise this callback over callbacks
            that actually use the selected organisations. This is done to ensure that a generic graph is visible
            whilst not being logged in by using a class object rather than callback itself
            """
            # only retrieve the organisation ids when the user is authenticated as it will otherwise break
            if authentication_status:
                self.Organisations_ids_to_query = [organisation['id'] for organisation in self.Organisations['data']
                                                   if organisation['name'] in organisation_to_include]

            return ""

        @self.App.callback(
            [Output('tab-pie-content', 'children'),
             Output('tab-bar-content', 'children')],
            Input('query-trigger-not-for-display', 'children'),
            #Input('tabs', 'value'),
            Input("dataset-variable", "value"))
        def render_content(query_trigger, dataset_variable):
            """
            :param any query_trigger: trigger to ensure that the callback is executed
            :param str dataset_variable: variable to be displayed in the pie chart
            :return:
            """
            # retrieve the data that is to be rendered
            filtered_data = self._retrieve_counts_to_render(dataset_variable)

            color_sequence = ["#4B0082", "#D8BFD8"]  # Two shades of purple

            # create a pie chart
            fig_pie = px.pie(filtered_data, names='Categories', values='Values',
                             color_discrete_sequence=color_sequence)
            fig_pie.update_layout(
                plot_bgcolor='lightgrey',
                paper_bgcolor='whitesmoke',
                font=dict(family=GLOBAL_FONT),
                title=dict(
                    text='Distribution of clinical variables',
                    y=0.92,  # position of the title
                    x=0.5,  # position of the title
                    xanchor='center',  # anchor the x position
                    yanchor='top',  # anchor the y position
                    font=dict(size=20, family=GLOBAL_FONT)  # font size of the title
                ),
            )
            pie_chart = dcc.Graph(figure=fig_pie, className='graph-shadow')

            bar_data = self._retrieve_scanners_to_render()

            # create a bar chart
            fig_bar = px.bar(bar_data, x='Scanners', y='Counts',
                             color_discrete_sequence=self.ColourSchemeCategorical)
            fig_bar.update_layout(
                plot_bgcolor='lightgrey',
                paper_bgcolor='whitesmoke',
                font=dict(family=GLOBAL_FONT),
                title=dict(
                    text='Scanner count',
                    y=0.92,  # position of the title
                    x=0.5,  # position of the title
                    xanchor='center',  # anchor the x position
                    yanchor='top',  # anchor the y position
                    font=dict(size=20, family=GLOBAL_FONT)  # font size of the title
                ),
            )

            bar_chart = dcc.Graph(figure=fig_bar, className='graph-shadow')

            return pie_chart, bar_chart

        # Define the mapping dictionary
        OrganisationIdToName = {2: 'HN1_Maastro', 3: 'Montreal', 4: 'Toronto', 5: 'HN3_Maastro', 7: 'HCG', 8: 'TMH'}

        # Define the callback function for the box plot
        @self.App.callback(
            Output('box-plot-content', 'children'),
            Input('query-trigger-not-for-display', 'children'),
            Input("box-plot-dropdown", "value"))
        def render_box_plot(query_trigger, box_plot_dropdown):
            """
            :param query_trigger: trigger to ensure that the callback is executed
            :param box_plot_dropdown: variable to be displayed in the box plot
            :return:
            """
            # Retrieve the data that is to be rendered
            if not self.Organisations_ids_to_query:
                box_plot_data = self._PlaceholderDataFeatures
                # Create a dummy  box plot
                fig_box = px.box(box_plot_data)

            else:
                box_data_dict = self._retrieve_features_to_render(box_plot_dropdown)
                # Create a DataFrame to hold the box plot data
                box_plot_data = pd.DataFrame(columns=['Value', 'Organization'])

                for org_id, stats_list in box_data_dict.items():
                    for stats in stats_list:
                        # Create a list of values that represent the box plot
                        values = [stats['min'], stats['q1'], stats['median'], stats['median'], stats['q3'], stats['max']]
                        values.extend(stats['outliers'])  # Add the outliers

                        # Create a DataFrame from these values
                        df = pd.DataFrame({
                            'Value': values,
                            'Organization': [org_id] * len(values)
                        })

                        # Append this DataFrame to the box plot data
                        box_plot_data = box_plot_data.append(df, ignore_index=True)

                # Normalize the 'Value' column
                box_plot_data['Value'] = (box_plot_data['Value'] - box_plot_data['Value'].min()) / (
                            box_plot_data['Value'].max() - box_plot_data['Value'].min())

                # Replace organization IDs with names
                box_plot_data['Organization'] = box_plot_data['Organization'].map(OrganisationIdToName)

                # Create a box plot
                fig_box = px.box(box_plot_data, x='Organization', y='Value')

                # Calculate median values for each organization
                ## Uncomment this code to add median annotations to the box plot
                #medians = box_plot_data.groupby('Organization')['Value'].median().round(2)

                # Add median annotations to the boxplot
                #for org, median in medians.items():
                #    fig_box.add_annotation(
                #        x=org,
                #        y=median,
                #        text=f'{median}',
                #        showarrow=False,
                #        font=dict(
                #            size=10,
                #            color="black"
                #        ),
                #       xanchor='center',
                #        yanchor='bottom',
                #        bgcolor='whitesmoke'
                #    )

            fig_box.update_layout(
                plot_bgcolor='lightgrey',
                paper_bgcolor='whitesmoke',
                font=dict(family=GLOBAL_FONT),
                title=dict(
                    text=f'Comparative box plots of Radiomic features across organizations - ({box_plot_dropdown})',
                    y=0.92,  # position of the title
                    x=0.5,  # position of the title
                    xanchor='center',  # anchor the x position
                    yanchor='top',  # anchor the y position
                    font=dict(size=20, family=GLOBAL_FONT)  # font size of the title
                ),
            )

            return dcc.Graph(figure=fig_box, className='graph-shadow')

        @self.App.callback(
            Output('heatmap-content', 'children'),
            Input('query-trigger-not-for-display', 'children'),
            Input("roi-checklist", "value"))
        def render_heatmap(query_trigger, roi_checklist):
            heatmap_data = self._retrieve_heatmap_to_render(roi_checklist)
            # cluster the data
            pairwise_distances = sch.distance.pdist(heatmap_data)
            linkage = sch.linkage(pairwise_distances, method='complete')
            cluster_distance_threshold = pairwise_distances.max() / 2
            idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold,
                                                criterion='distance')
            idx = np.argsort(idx_to_cluster_array)
            # reorder the data
            if isinstance(heatmap_data, pd.DataFrame):
                heatmap_data_ordered = heatmap_data.iloc[idx, :].T.iloc[idx, :]
            heatmap_data_ordered = heatmap_data.iloc[idx, :].iloc[:, idx]

            # Custom color scale
            color_scale = ["rgb(0,0,255)", "rgb(211,211,211)", "rgb(255,0,0)"]  # blue, grey, red
            fig_heatmap = px.imshow(heatmap_data_ordered, y=heatmap_data_ordered.columns, text_auto=False,
                                    aspect="auto",
                                    title='Correlation Heatmap', color_continuous_scale=color_scale)

            # Increase the title font size to 24 and the y-axis labels font size to 12
            fig_heatmap.update_layout(
                coloraxis_colorbar=dict(
                    tickvals=[-1, 0, 1],  # specify the values
                    ticktext=['-1', '0', '1'],  # specify the labels
                    len=0.5,  # specify the length of the color bar as a fraction of the plot area
                    yanchor="top",  # anchor the color bar at the top
                    y=1  # position the color bar at the top of the plot area
                ),
                title=dict(
                    text='Correlation heatmap of Radiomic features',
                    y=0.92,  # position of the title
                    x=0.5,  # position of the title
                    xanchor='center',  # anchor the x position
                    yanchor='top',  # anchor the y position
                    font=dict(size=20, family=GLOBAL_FONT)  # font size of the title
                ),
                font=dict(family=GLOBAL_FONT)
            )

            fig_heatmap.update_yaxes(
                tickfont=dict(size=7),  # font size of the y-axis labels
            )
            # Increase the x-axis labels font size to 12
            fig_heatmap.update_xaxes(
                tickfont=dict(size=7),  # font size of the x-axis labels
            )
            fig_heatmap.update_layout(
                plot_bgcolor='lightgrey',
                paper_bgcolor='whitesmoke',
                font=dict(family=GLOBAL_FONT)
            )
            #return dcc.Graph(figure=fig_heatmap, style={'height': '800px', 'width': '100%'}, className='graph-shadow')
            return dcc.Graph(figure=fig_heatmap, style={'height': '800px', 'width': '100%'})

    def run(self, debug=None, port=8050):
        """
        Start the Plotly Dash dashboard

        :param bool debug: specify whether to use debugging mode
        """
        if isinstance(debug, bool) is False:
            debug = True

        self.App.run_server(debug=debug, port=port, dev_tools_ui=False)

    def _retrieve_counts_to_render(self, dataset_variable):
        """
        Retrieve counts of given variable, either from data already existing in the placeholder, or by querying Vantage6

        :param str dataset_variable: name or predicate of the variable to query
        :return: pandas.DataFrame consisting of the counts of the desired variable
        """
        if not self.Organisations_ids_to_query:
            filtered_data = self.PlaceholderDataframe[
                self.PlaceholderDataframe['HashIdentifier'] == miscellaneous.hash_information('Not an actual variable',
                                                                                              {}, [])]
            return filtered_data

        if dataset_variable != 'Not an actual variable':
            self.Filters_to_apply = {dataset_variable: self.filter_dict[dataset_variable]}

        # do not attempt to query dummy data; the [] and [0] represent the default filter and organisation state
        if f'{dataset_variable}_count' in self._PlaceholderData.keys():
            filtered_data = self.PlaceholderDataframe[
                self.PlaceholderDataframe['HashIdentifier'] == miscellaneous.hash_information(dataset_variable,
                                                                                              {}, [])]
            return filtered_data
        else:
            # collect the data that is to be queried
            filtered_data = self.PlaceholderDataframe[self.PlaceholderDataframe['HashIdentifier'] ==
                                                      miscellaneous.hash_information(dataset_variable,
                                                                                     self.Filters_to_apply,
                                                                                     self.Organisations_ids_to_query)]

            # if it appears empty, i.e., the data for the organisation hash is not available then retrieve it
            if filtered_data.empty:
                query_name = f'Dashboard request of counts for {dataset_variable}'
                self.Vantage6User.compute_count_sparql(name=query_name,
                                                       predicates=dataset_variable,
                                                       organisation_ids=self.Organisations_ids_to_query,
                                                       filters=self.Filters_to_apply,
                                                       save_results=False)
                query_result = self.Vantage6User.Results[query_name]

                # append the query result
                self.PlaceholderDataframe = miscellaneous.convert_count_dict_to_dataframe(query_result,
                                                                                          self.Filters_to_apply,
                                                                                          self.Organisations_ids_to_query,
                                                                                          self.PlaceholderDataframe)

            # filter the placeholder to retrieve the new data
            filtered_data = self.PlaceholderDataframe[self.PlaceholderDataframe['HashIdentifier'] ==
                                                      miscellaneous.hash_information(dataset_variable,
                                                                                     self.Filters_to_apply,
                                                                                     self.Organisations_ids_to_query)]
            for col in filtered_data.columns:
                filtered_data[col] = filtered_data[col].map(self.codedict).fillna(filtered_data[col])

            return filtered_data

    def _retrieve_scanners_to_render(self):
        """
        Retrieve counts of given variable, either from data already existing in the placeholder, or by querying Vantage6

        :param str dataset_variable: name or predicate of the variable to query
        :return: pandas.DataFrame consisting of the counts of the desired variable
        """

        # do not attempt to query dummy data; the [] and [0] represent the default filter and organisation state
        # collect the data that is to be queried
        bar_data = self.PlaceholderDataframeScanners[self.PlaceholderDataframeScanners['HashIdentifier'] ==
                                                     miscellaneous.hash_information(self.Organisations_ids_to_query)]

        # if it appears empty, i.e., the data for the organisation hash is not available then retrieve it
        if bar_data.empty:
            query_name = f'Dashboard request of counts for scanners'
            self.Vantage6User.compute_scanner_sparql(name=query_name,
                                                     organisation_ids=self.Organisations_ids_to_query,
                                                     save_results=False)
            query_result = self.Vantage6User.Results[query_name]

            # append the query result
            self.PlaceholderDataframeScanners = miscellaneous.convert_scan_dict_to_dataframe(query_result,
                                                                                             self.Organisations_ids_to_query,
                                                                                             self.PlaceholderDataframeScanners)

        # filter the placeholder to retrieve the new data
        bar_data = self.PlaceholderDataframeScanners[self.PlaceholderDataframeScanners['HashIdentifier'] ==
                                                     miscellaneous.hash_information(self.Organisations_ids_to_query)]
        return bar_data

    def _retrieve_features_to_render(self, box_plot_dropdown):
        """
        Retrieve counts of given variable, either from data already existing in the placeholder, or by querying Vantage6

        :param str dataset_variable: name or predicate of the variable to query
        :return: pandas.DataFrame consisting of the counts of the desired variable
        """

        # do not attempt to query dummy data; the [] and [0] represent the default filter and organisation state
        # collect the data that is to be queried
        box_data = {}
        for org_id in self.Organisations_ids_to_query:
            query_name = f'Dashboard request of radiomic features for {org_id}'
            self.Vantage6User.compute_features_sparql(name=query_name,
                                                      feature=box_plot_dropdown,
                                                      organisation_ids=[org_id],
                                                      save_results=False)
            box_data[org_id] = self.Vantage6User.Results[query_name]
        return box_data

    def _retrieve_heatmap_to_render(self, roi_checklist):
        """
        Retrieve either from data already existing in the placeholder, or by querying Vantage6

        :return: pandas.DataFrame consisting of the counts of the desired variable
        """
        # build in a check for the filter or alike thing, to ensure that it is not directly querying data
        # collect the data that is to be queried
        expl_vars = []
        censor_col = 'censor'

        # if organizations ids are selected, check the hash id if already present and fetch it
        if self.Organisations_ids_to_query:
            OrganisationHash = hashlib.sha256(
                (str(tuple(self.Organisations_ids_to_query)) + roi_checklist).encode()).hexdigest()
        # else get the default hash id with organization ids [] and roi filter as GTV-1
        else:
            OrganisationHash = hashlib.sha256(
                (str(tuple(self.Organisations_ids_to_query)) + 'C100346').encode()).hexdigest()

        if OrganisationHash in self._PlaceholderDataHeatMap:
            columns_to_fetch = self._PlaceholderDataHeatMap[OrganisationHash]['columns']
            roi_to_fetch = self._PlaceholderDataHeatMap[OrganisationHash]['ROI']

            heatmap_data = self.PlaceholderDataFrameHeatMap[columns_to_fetch]
            heatmap_data = heatmap_data[(self.PlaceholderDataFrameHeatMap['ROI'] == roi_to_fetch) &
                                        (self.PlaceholderDataFrameHeatMap['OrganisationHash'] == OrganisationHash)]
            heatmap_data = heatmap_data.drop(columns=['ROI', 'OrganisationHash'])

            return heatmap_data

        else:
            query_name = f'Heatmap for {self.Organisations_ids_to_query} with filter ROI'

            # use right task
            self.Vantage6User.compute_hm_sparql(name=query_name,
                                                expl_vars=expl_vars,
                                                censor_col=censor_col,
                                                roitype=roi_checklist,
                                                organisation_ids=self.Organisations_ids_to_query,
                                                save_results=False)

            query_result = self.Vantage6User.Results[query_name]

            # append the query result
            self.PlaceholderDataFrameHeatMap, temp_dict = miscellaneous.convert_heatmap_to_appropriate_dataframe(
                query_result,
                self.Organisations_ids_to_query,
                roi_checklist,
                self.PlaceholderDataFrameHeatMap)

            self._PlaceholderDataHeatMap.update(temp_dict)

            columns_to_fetch = self._PlaceholderDataHeatMap[OrganisationHash]['columns']
            roi_to_fetch = self._PlaceholderDataHeatMap[OrganisationHash]['ROI']

            # collect the data that is to be queried
            heatmap_data = self.PlaceholderDataFrameHeatMap[columns_to_fetch]
            heatmap_data = heatmap_data[(self.PlaceholderDataFrameHeatMap['ROI'] == roi_to_fetch) &
                                        (self.PlaceholderDataFrameHeatMap['OrganisationHash'] == OrganisationHash)]
            heatmap_data = heatmap_data.drop(columns=['ROI', 'OrganisationHash'])

        return heatmap_data


if __name__ == '__main__':
    dash_app = Dashboard()
    dash_app.run(port=8052)
