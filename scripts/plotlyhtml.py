import csv,gc  # garbage memory collection :)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import jgraph as ig

import plotly
import plotly.plotly as py
import plotly.graph_objs as go

## Type in the path to your csv file here
thedata = np.genfromtxt('../csv/Fear199biglocaleq.brightest.csv', delimiter=',', dtype='int', usecols = (0,1,2), names=['a','b','c'])

trace1 = go.Scatter3d(
    x = thedata['a'],
    y = thedata['b'],
    z = thedata['c'],
    mode='lines',
    line=dict(
    width= 0,
    colorscale= "Viridis",
    opacity=0.1
    )
    #marker=dict(
     #   size=1.2,
    #    color='purple',                # set color to an array/list of desired values
    #    colorscale='Viridis',   # choose a colorscale
    #    opacity=0.15
    #)
)

data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
    
fig = go.Figure(data=data, layout=layout)
print "localeq"
plotly.offline.plot(fig, filename= "linesopaque")
