from vantage6.tools.mock_client import ClientMockProtocol
import vtg_corr  

# Initialize the mock server. The datasets simulate the local datasets from
# the node. In this case we have two parties having two different datasets:
# a.csv and b.csv. The module name needs to be the name of your algorithm
# package. This is the name you specified in `setup.py`, in our case that
# would be v6-correlation-matrix-py.
client = ClientMockProtocol(
    datasets=["/home/rmucsc.rm.unicatt.it/sb001956/TOTALRadiomics/lang_shift/vtg_set_up/vtg_set_up/local/TOTAL_Radiomics_gemelli.csv",
              "/home/rmucsc.rm.unicatt.it/sb001956/TOTALRadiomics/lang_shift/vtg_set_up/vtg_set_up/local/TOTAL_Radiomics_gemelli.csv"],
    module="vtg_cfs"
)

# to inspect which organization are in your mock client, you can run the
# following
#organizations = client.get_organizations_in_my_collaboration()
org_ids = ids = [1,2]



# To trigger the master method you also need to supply the `master`-flag
# to the input. Also note that we only supply the task to a single organization
# as we only want to execute the central part of the algorithm once. The master
# task takes care of the distribution to the other parties.
average_task = client.create_new_task(
    input_={
        'master': 1,
        'method':'master',
        'kwargs': { #'test_size': 0.2,
            'expl_vars': ['Fcm.auto.corr',
                         'Fcm.clust.prom',
                         'Fcm.clust.shade',
                         'Fcm.clust.tend',
                         'Fcm.contrast',
                         'Fcm.corr',
                         'Fcm.diff.avg',
                         'Fcm.diff.entr',
                         'Fcm.diff.var',
                         'Fcm.energy',
                         'Fcm.info.corr.1',
                         'Fcm.info.corr.2',
                         'Fcm.inv.diff']
            ,'censor_col':'C', 'data_set':'all'
                                }
    },
    organization_ids=[org_ids[0]]
)

results = client.get_results(average_task.get("id"))
results[0]['correlationMatrix'].to_csv("correlation_matrix.csv")
print(results)
