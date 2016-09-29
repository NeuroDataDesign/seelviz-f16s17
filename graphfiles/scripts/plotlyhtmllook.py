from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import *
from plotly import tools

import clearity as cl  # I wrote this module for easier operations on data
import clearity.resources as rs
import csv,gc  # garbage memory collection :)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import jgraph as ig

from plotly.offline import download_plotlyjs, plot
from plotly.graph_objs import *
from plotly import tools
import plotly
#plotly.tools.set_credentials_file(username='lukezhu1', api_key='rkacjaly2k')

thedata = np.genfromtxt('/home/albert/Thumbo/csv/Aut1367localeq.40000.brightest.csv', delimiter=',', dtype='int', usecols = (0,1,2), names=['a','b','c'])

trace1 = Scatter3d(
    x = thedata['a'],
    y = thedata['b'],
    z = thedata['c'],
    mode='markers',
    marker=dict(
        size=1.2,
        color='purple',                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.15
    )
)

data = [trace1]
layout = Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
    
fig = Figure(data=data, layout=layout)
print "localeq"
plotly.offline.plot(fig, filename= "/home/albert/Thumbo/html/Aut1367localeq.40000.brightest.csv")
