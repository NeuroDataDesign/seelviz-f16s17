#!/usr/bin/env python
#-*- coding:utf-8 -*-

import matplotlib
from matplotlib import pyplot as plt

import numpy as np
from numpy import linalg as LA
import cv2
import math

import plotly
from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly import tools

import time
import collections as col
from collections import OrderedDict

import ast
from ndreg import *
import ndio.remote.neurodata as neurodata

import nibabel as nib
import networkx as nx
import re
import pandas as pd
import requests
import json
import seaborn as sns
import csv, gc

from sklearn.manifold import spectral_embedding as se

import scipy.sparse as sp

plotly.offline.init_notebook_mode()

def spec_clust(graphx, num_components):
    """
    Function for doing the spectral embedding.
    :param graphx:
    :param num_components:
    :return:
    """
    adj_mat = nx.adjacency_matrix(graphx)
#     result = se(adj_mat, n_components=num_components, drop_first=True)
    result = se(adj_mat, n_components=num_components, drop_first=False)

    return result


def add_to_dict(d, region, index):
    if region in d:
        d[region].append(index)
    else:
        d[region] = [index]

    return d


def get_adj_mat(regions_path):
    points = np.genfromtxt(regions_path, delimiter=',')
    x_dim = np.max(points[:, 0])
    y_dim = np.max(points[:, 1])
    z_dim = np.max(points[:, 2])
    am = SparseMatrix(x_dim, y_dim, z_dim)
    for point in points:
        am.add(tuple(point[0:3]), point[4])

    return am


def get_dict_real(g, se_result, regions_path):
    nodes = g.nodes()
    points = np.genfromtxt(regions_path, delimiter=',')
    orig_dict = OrderedDict()
    d = {}

    sparse_mat = get_adj_mat(regions_path)

    for index, node in enumerate(nodes):
        s = g.node[node]['attr']
        point = ast.literal_eval(s)
        region = sparse_mat.get(tuple(point))
        #         if region == -1:
        #             # error
        #             print 'FUCK'
        add_to_dict(d, region, index)

    for point in points:
        region = point[4]
        #         if region in orig_dict:
        #             orig_dict[region] = np.vstack((orig_dict[region], point[0:3]))
        #         else:
        #             orig_dict[region] = np.array([point[0:3]])
        add_to_dict(orig_dict, region, point[0:3])

    se_regions_nodes = {}
    se_regions = {}
    for key, value in d.iteritems():
        index_list = value
        nodes_arr = np.array(nodes)
        pt_names = nodes_arr[index_list]
        se_pts = se_result[index_list]
        nodes_to_se = dict(zip(pt_names, se_pts))  # maps from node names to embedded point coordinates
        se_regions_nodes[key] = nodes_to_se
        se_regions[key] = se_pts

    return se_regions, orig_dict, se_regions_nodes


def create_connectivity_graph(orig_avg_dict, se_avg_dict, max_dist=0.02):
    g = nx.Graph()
    for key, avg in se_avg_dict.iteritems():
        for key2, avg2 in se_avg_dict.iteritems():
            avg_np = np.array(avg)
            avg2_np = np.array(avg2)
            diff = np.linalg.norm(avg_np - avg2_np)
            diff = max_dist if diff > max_dist else diff
            g.add_edge(key, key2, weight=diff)

    # Setting the coordinate attribute for each region node to the average of that region.
    for key, avg in orig_avg_dict.iteritems():
        g.node[key]['attr'] = avg

    return g

def get_connectivity_hard(eig_dict, orig_dict=None, max_dist=0.02):
    """
    Uses create_connectivity_graph.
    :param eig_dict:
    :param orig_dict:
    :return:
    """
    eigenvector_index = 1  # the second smallest eigenvector
    avg_dict = {}
    orig_avg_dict = OrderedDict()

    # dict that maps from region to most connected region
    con_dict = OrderedDict()

    orig_con_dict = OrderedDict()

    if orig_dict != None:
        # Getting the original averages.
        for key, region in orig_dict.iteritems():
            tmp_x = []
            tmp_y = []
            y_vals = []

            for j in range(len(region)):
                y_vals.append(region[j])
            y_vals = np.array(y_vals)
            x_avg = np.mean(y_vals[:, 0])
            y_avg = np.mean(y_vals[:, 1])
            z_avg = np.mean(y_vals[:, 2])
            orig_avg_dict[key] = [x_avg, y_avg, z_avg]
        # avg = np.mean(y_vals)
        #             orig_avg_dict[key] = avg

        #         print 'orignal averages'
        #         print orig_avg_dict

        # Getting connectivity for original points.
        for key, avg in orig_avg_dict.iteritems():
            min_key = ''
            min_diff = float('inf')
            for key2, avg2 in orig_avg_dict.iteritems():
                if key2 == key:
                    continue
                avg_np = np.array(avg)
                avg2_np = np.array(avg2)
                diff = np.linalg.norm(avg_np - avg2_np)
                if diff < min_diff:
                    min_diff = diff
                    min_key = key2

            orig_con_dict[float(key)] = [float(min_key), min_diff]

    # Getting the average first 2 eigenvector components for each of the regions
    for key, region in eig_dict.iteritems():
        #         print(key)
        y_vals = []

        for j in range(len(region)):
            y_vals.append(region[j])
        y_vals = np.array(y_vals)
        x_avg = np.mean(y_vals[:, 0])
        y_avg = np.mean(y_vals[:, 1])
        z_avg = np.mean(y_vals[:, 2])
        avg_dict[key] = [x_avg, y_avg, z_avg]

    # print('getcon avg_dict')
    #     print(avg_dict)

    # Computing connectivity between regions using the distance between averages
    for key, avg in avg_dict.iteritems():
        min_key = ''
        min_diff = float('inf')
        for key2, avg2 in avg_dict.iteritems():
            if key2 == key:
                continue
            avg_np = np.array(avg)
            avg2_np = np.array(avg2)
            diff = np.linalg.norm(avg_np - avg2_np)
            if diff < min_diff:
                min_diff = diff
                min_key = key2

        con_dict[float(key)] = [float(min_key), min_diff]

    con_dict = OrderedDict(sorted(con_dict.items()))
    orig_con_dict = OrderedDict(sorted(orig_con_dict.items()))

    g = create_connectivity_graph(orig_avg_dict, avg_dict, max_dist)

    if orig_dict == None:
        return con_dict
    else:
        return con_dict, orig_con_dict, g

class SparseMatrix:
    def __init__(self, x, y, z):
#         self._max_index = 0
        x_dim = x
        y_dim = y
        z_dim = z
        self._vector = {}

    def add(self, index, value):
        # vector starts at index one, because it reads from the file and the file
        # always has the index of the features start at 1
        self._vector[index] = value
#         if index > self._max_index:
#             self._max_index = index

    def get(self, index):
        # if the index doesn't exist in the dict, return 0 because it's sparse anyways
        if index in self._vector:
            return self._vector[index]
        return -1

    def get_sparse_matrix(self):
        return self._vector
        # return self._vector.keys()

#     def get_full_vector(self, size=None):
#         """ Returns a full vector of features as a numpy array. """
#         size = (self._max_index + 1) if size == None else size
#         full_vector = np.zeros(size)  # 0 indexed
#         for key, value in self._vector.iteritems():
#             full_vector[key] = value

#         return full_vector

    def __str__(self):
        return str(self._vector)

def plot_con_mat(con_adj_mat, output_path=None, show=False):
    title = 'Connectivity Heatmap'
    data = [
        Heatmap(
            z = con_adj_mat,
    #         x = con_graph.nodes(),
    #         y = con_graph.nodes()
        )
    ]
    layout = Layout(
        title = title,
        xaxis=dict(title='region'),
        yaxis=dict(title='region')
    )
    fig = Figure(data=data, layout=layout)
    if show:
        iplot(fig)
    if output_path != None:
        plotly.offline.plot(fig, filename=output_path)

    return fig