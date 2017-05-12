from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import *
from plotly import tools
import plotly

import numpy as np
from numpy import linalg as LA
from sklearn.manifold import spectral_embedding as se

import re
import matplotlib
import seaborn as sns
import networkx as nx
import math

def plot_connectivity(dictionary, outfile):
    ## plot points in XY-plane
    ## clusters of points are determined beforehand, each group given a different color
    ## also plots centroid of each cluster ((x,y) of centroid is average of cluster points)
    ## lines indicate nearest centroid by Euclidean distance, prints distance
    n = len(dictionary.keys())
    current_palette = sns.color_palette("husl", n)
    Xe = []
    Ye = []
    Ze = []
    data = []
    avg_dict = OrderedDict()
    i = 0
    for key, region in dictionary.iteritems():
        X = []
        Y = []
        Z = []
        tmp_x = []
        tmp_y = []
        tmp_z = []
        region_col = current_palette[i]
        region_col_lit = 'rgb' + str(region_col)
        i += 1
        for coord in region:    
            X.append(coord[0])
            Y.append(coord[1])
            Z.append(coord[2])
            tmp_x.append(coord[0])
            tmp_y.append(coord[1])
            tmp_z.append(coord[2])
        avg_dict[key] = [[np.mean(tmp_x), np.mean(tmp_y), np.mean(tmp_z)]]
            
        trace_scatter = Scatter3d(
                x = X, 
                y = Y,
                z = Z,
                name=key,
                mode='markers',
                marker=dict(
                    size=10,
                    color=region_col_lit, #'purple',                # set color to an array/list of desired values
                    colorscale='Viridis',   # choose a colorscale
                    opacity=0.1
                )
        )
        avg_scatter = Scatter3d(
                x = [avg_dict[key][0][0]],
                y = [avg_dict[key][0][1]],
                z = [avg_dict[key][0][2]],
                mode='markers',
                name=key+'_avg',
                marker=dict(
                    size=10,
                    color=region_col_lit,
                    colorscale='Viridis',
                    line=dict(
                        width = 2,
                        color = 'rgb(0, 0, 0)'
                    )
                )
        )
        data.append(trace_scatter)
        data.append(avg_scatter)
        
    connectivity=np.zeros((n,n))
    locations = avg_dict.keys()
    for i, key in enumerate(avg_dict):
        tmp = []
        for j in range(len(locations)):
            if j == i:
                connectivity[i,j] = 1
#                 connectivity2[i,j] = 1
                continue
            p1 = np.asarray(avg_dict[key][0])
            p2 = np.asarray(avg_dict[locations[j]][0])
            dist = LA.norm(p1 - p2)
            tmp.append(dist)
            connectivity[i,j] = math.exp(-dist*100)
#             connectivity2[i,j] = math.exp(-(dist)**2)
#             print "Distance between region " + key + " and region " + locations[j] + " is: " + str(dist)
        newmin = tmp.index(min(tmp))
        if newmin >= i:
            newmin += 1
#         print "region " + key + " is closest to region " + locations[newmin] + "\n"
        tmp2 = avg_dict.keys()[newmin]
        Xe+=[avg_dict[key][0][0],avg_dict[tmp2][0][0],None]
        Ye+=[avg_dict[key][0][1],avg_dict[tmp2][0][1],None]
        Ze+=[dictionary[key][0][2],dictionary[tmp2][0][2],None]
    
    trace_edge = Scatter3d(x=Xe,
               y=Ye,
               z=Ze,
               mode='lines',
               line=Line(color='rgb(0,0,0)', width=3),
               hoverinfo='none'
    )

    data.append(trace_edge)
    
    layout = Layout(
        paper_bgcolor='rgb(255,255,255)',
        plot_bgcolor='rgb(255,255,255)'
    )
        
    fig = Figure(data=data, layout=layout)
    # iplot(fig, validate=False)
    plotly.offline.plot(fig, filename=outfile)
    return connectivity


def connectivity_heatmap(b_hat, outdir):
    trace = Heatmap(z=b_hat)
    data = [trace]
    layout = Layout(title='Spectral Embedding Estimated Connectivity B_hat')
    fig = Figure(data=data, layout=layout)
    # iplot(fig, validate=False)
    plotly.offline.plot(fig, filename=outdir + "\\heatmap_est_connectivity.html")


def estimate_connectivity(graphml, outdir):
    G = nx.read_graphml(graphml)

    # generate adjacency matrix from networkx after removing edgeless nodes
    # scipy sparse matrix
    outdeg = G.degree()
    to_keep = [n for n in outdeg if outdeg[n] != 0]
    H = G.subgraph(to_keep)
    A2 = nx.adjacency_matrix(H)

    # use sklearn's implementation of spectral_embedding to calculate
    # laplacian and obtain eigenvectors and eigenvalues from it
    a2out = se(A2,n_components=3,drop_first=True)

    nodelist = H.nodes()
    se_regions = {}
    for i, node in enumerate(nodelist):
        reg = H.node[node]['region']
        pos = a2out[i]
        if str(reg) not in se_regions:
            se_regions[str(reg)] = [pos]
        else:
            se_regions[str(reg)].append(pos)

    connect = plot_connectivity(se_regions, outdir + "\\spectral_embedding.html")
    connectivity_heatmap(connect, outdir)
    return connect
