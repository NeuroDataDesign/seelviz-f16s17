from plotly.offline import download_plotlyjs
from plotly.graph_objs import *
from plotly import tools
import plotly

import os
os.chdir('C:/Users/L/Documents/Homework/BME/Neuro Data I/Data/')

import csv,gc  # garbage memory collection :)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from mpl_toolkits.mplot3d import axes3d
from collections import namedtuple

import csv
import re
import matplotlib
import time
import seaborn as sns

font = {'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)

### load data
thedata = np.genfromtxt('./Control258localeq.region.csv', delimiter=',', dtype='int', usecols = (0,1,2,4), names=['x','y','z','region']) 
###

from collections import OrderedDict
region_dict = OrderedDict()
for l in thedata:
    trace = 'trace' + str(l[3])
    if trace not in region_dict:
        region_dict[trace] = np.array([[l[0], l[1], l[2], l[3]]])
    else:
        tmp = np.array([[l[0], l[1], l[2], l[3]]])
        region_dict[trace] = np.concatenate((region_dict.get(trace, np.zeros((1,4))), tmp), axis=0)

current_palette = sns.color_palette("husl", 396)
# print current_palette

data = []
for i, key in enumerate(region_dict):
    trace = region_dict[key]
    tmp_col = current_palette[i]
    tmp_col_lit = 'rgb' + str(tmp_col)
    trace_scatter = Scatter3d(
        x = trace[:,0], 
        y = trace[:,1],
        z = trace[:,2],
        mode='markers',
        marker=dict(
            size=1.2,
            color=tmp_col_lit, #'purple',                # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            opacity=0.15
        )
    )
    
    data.append(trace_scatter)
    

layout = Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ),
    paper_bgcolor='rgb(0,0,0)',
    plot_bgcolor='rgb(0,0,0)'
)

fig = Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename= "Control258localeq.region_color.html")