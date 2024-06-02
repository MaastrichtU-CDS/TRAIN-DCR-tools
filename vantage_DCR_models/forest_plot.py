import pandas as pd
import matplotlib.pyplot as plt
import forestplot as fp

# Load CSV files containing results
df_clin = pd.read_csv("result_df_OS_clin.csv")
df_rad_prim = pd.read_csv("result_df_OS_prim.csv")
df_rad_node = pd.read_csv("result_df_OS_node.csv")
df_combined = pd.read_csv("result_df_OS_lp.csv")

# Rename feature name in combined results
df_combined.loc[df_combined['OS'] == 'lp_Clinical_all', 'OS'] = 'lp_Clinical'

# Add a 'source' column to each DataFrame to indicate which CSV file the data came from
df_clin['source'] = 'Clinical'
df_rad_prim['source'] = 'Radiomics_primary'
df_rad_node['source'] = 'Radiomics_node'
df_combined['source'] = 'Combined'

# Concatenate the DataFrames
df_all = pd.concat([df_clin, df_rad_prim, df_rad_node, df_combined])
# Add a new column 'est_ci'
df_all['est_ci'] = df_all.apply(lambda row: f"{row['Exp(coef)']} ({row['lower_CI']} to {row['upper_CI']})", axis=1)

# Create forest plot
ax = fp.forestplot(df_all,  # the dataframe with results data
              estimate="Exp(coef)",  # col containing estimated effect size
              ll="lower_CI", hl="upper_CI",  # columns containing conf. int. lower and higher limits
              varlabel="OS",  # column containing variable label
              annote=["est_ci"],  # columns to report on left of plot
              annoteheaders=["Hazard ratio (95% Conf. Int.)"],  # corresponding headers
              xlabel="Overall Survival - Loop 2",  # x-label title
              groupvar="source",  # Add variable groupings
              # group ordering
              group_order=["Clinical", "Radiomics_primary", "Radiomics_node", "Combined"],
              sort=True,  # sort in ascending order (sorts within group if group is specified)
              table=True,  # Format as a table
              pval="p-value",  # Column of p-value to be reported on right
              color_alt_rows=True,  # Gray alternate rows
              logscale=False,
              #xticks=[-.4, -.2, 0, .2],  # x-ticks to be printed
              # Additional kwargs for customizations
              **{"marker": "D",  # set maker symbol as diamond
                 "markersize": 35,  # adjust marker size
                 "xlinestyle": (0, (10, 5)),  # long dash for x-reference line
                 "xlinecolor": "#808080",  # gray color for x-reference line
                 "xtick_size": 12,  # adjust x-ticker fontsize
                }
              )

# Add dotted line at x = 1.0
ax.axvline(x=1.0, linestyle='--', color='gray')

# Save the plot
plt.savefig("plot_OS_loop2.png", bbox_inches="tight")