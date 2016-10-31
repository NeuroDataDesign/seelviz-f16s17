#!/usr/bin/env python
#-*- coding:utf-8 -*-
from __future__ import print_function

__author__ = 'seelviz'

from plotly.offline import download_plotlyjs
from plotly.graph_objs import *
from plotly import tools
import plotly

import os
#os.chdir('C:/Users/L/Documents/Homework/BME/Neuro Data I/Data/')

import csv,gc  # garbage memory collection :)

import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import axes3d

# from mpl_toolkits.mplot3d import axes3d
# from collections import namedtuple

import csv
import re
import matplotlib
import time
import seaborn as sns

from collections import OrderedDict

class atlasregiongraph(object):
    """Class for generating the color coded atlas region graphs"""

    def __init__(self, token, path=None):
        self._token = token
        self._path = path
        data_txt = ""
        if path == None:
            data_txt = token + '/' + token + '.csv'
        else:
            data_txt = path + '/' + token + '.csv'
        self._data = np.genfromtxt(data_txt, delimiter=',', dtype='int', usecols = (0,1,2,4), names=['x','y','z','region'])

    def generate_atlas_region_graph(self, path=None, numRegions = 10):
        font = {'weight' : 'bold',
            'size'   : 18}

        matplotlib.rc('font', **font)
        thedata = self._data
        if path == None:
            thedata = self._data
        else:
        ### load data
            thedata = np.genfromtxt(self._token + '/' + self._token + '.csv', delimiter=',', dtype='int', usecols = (0,1,2,4), names=['x','y','z','region'])

        region_dict = OrderedDict()
        for l in thedata:
            trace = 'trace' + str(l[3])
            if trace not in region_dict:
                region_dict[trace] = np.array([[l[0], l[1], l[2], l[3]]])
            else:
                tmp = np.array([[l[0], l[1], l[2], l[3]]])
                region_dict[trace] = np.concatenate((region_dict.get(trace, np.zeros((1,4))), tmp), axis=0)

        current_palette = sns.color_palette("husl", numRegions)
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
        plotly.offline.plot(fig, filename= self._path + '/' + self._token + "_region_color.html")

