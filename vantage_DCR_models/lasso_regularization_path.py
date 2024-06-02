import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio


### REGULARIZATION PATH ####
# Read the CSV file with Lasso results
 
df = pd.read_csv('lasso_path_DM.csv')
 
df2 = df.T
df2.columns = df2.iloc[0]
df2 = df2.drop(df2.index[0])
df2.index = [str(round(el,4)) for el in df2.index.astype(float)]
 
 
# color = ['#d7191c','#fdae61','#ffffbf','#abd9e9','#2c7bb6']
color = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf',  '#984ea3', '#999999', '#e41a1c', '#dede00']
 
marks = [
 'o',
 'v',
 '^',
 '<',
 '>',
 '*',
 '+',
 'x',
 'D',
 'X']
 
layout = go.Layout( paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    title='Regularization path of clinical variables for DM (Loop 2)',
    xaxis=dict(title='Regularization parameter ($\lambda$)'),
    yaxis=dict(title='Coefficients'),
    font=dict(family='Times New Roman', size=16),
    xaxis2= {'anchor': 'y', 'overlaying': 'x', 'side': 'top', 'tickvals': df2.index,
                'ticktext': df.replace(0, np.nan).count().to_list()[1:]}
)
 
data = []
for i in range(len(list(df2))):
  data.append(go.Scatter(x=df2.index, y=df2[list(df2)[i]], mode='lines', name=list(df2)[i], line_color=color[i]))
fig = go.Figure(data=data, layout=layout)
 
fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True,ticks='outside', showgrid = True)
fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True,ticks='outside' )
fig.update_layout(title_x=0.5, legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))
fig.data[1].update(xaxis='x2')
config = {
    'toImageButtonOptions': {
        'format': 'png',  # one of png, svg, jpeg, webp
        'filename': 'lasso_clin_DM_loop2',
        'height': 600,
        'width': 1000,
        'scale': 3  # Multiply title/legend/axis/canvas sizes by this factor
    }
}
fig.show(config=config)
#fig.show()