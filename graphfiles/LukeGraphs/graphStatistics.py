from argparse import ArgumentParser
from collections import OrderedDict
from subprocess import Popen
from scipy.stats import gaussian_kde

import numpy as np
import nibabel as nb
import networkx as nx
import os
import pickle

filename = '/Users/albertlee/seelviz/graphfiles/LukeGraphs/Fear199localeq.graphml'
outfile = '/Users/albertlee/seelviz/graphfiles/LukeGraphs/'


#def write(outdir, metric, data):
#    """
#    Write computed derivative to disk in a pickle file

#    Required parameters:
#        outdir:
#            - Path to derivative save location
#        metric:
#            - The value that was calculated
#        data:
#            - The results of this calculation
#        :
#            - Name of  of interest as it appears in the directory titles
#    """
#    of = open(outdir + '/' + '_' + metric + '.pkl', 'wb')
#    pickle.dump({metric: data}, of)
#    of.close()

def loadGraphs(filename, verb=False):
    """
    Given a list of files, returns a dictionary of graphs

    Required parameters:
        filenames:
            - List of filenames for graphs
    Optional parameters:
        verb:
            - Toggles verbose output statements
    """
    #  Initializes empty dictionary
    gstruct = OrderedDict()
    for idx, files in enumerate(filename):
        if verb:
            print "Loading: " + filename
        #  Adds graphs to dictionary with key being filename
        fname = os.path.basename(filename)
        gstruct[fname] = nx.read_graphml(filename)
    return gstruct

def constructGraphDict(filename, verb=False):
    """
    Given a set of files and a directory to put things, loads graphs.

    Required parameters:
        names:
            - List of names of the datasets
        fs:
            - Dictionary of lists of files in each dataset
    Optional parameters:
        verb:
            - Toggles verbose output statements
    """
    #  Loads graphs into memory for all datasets
    graphs = OrderedDict()
    for idx, name in enumerate(filename):
        if verb:
            print "Loading Dataset: " + name
        # The key for the dictionary of graphs is the dataset name
        graphs[name] = loadGraphs(filename)
    return graphs

#print len(nx.edges(graphs[filename][subj]))
graphs = constructGraphDict(filename)
print "Computing: NNZ"
nnz = OrderedDict()
for idx, name in enumerate(filename):
    nnz[name] = OrderedDict((subj, len(nx.edges(graphs[name][subj])))
                            for subj in graphs[name])
print nnz[name]

#print "Computing: NNZ"
#nnz = OrderedDict()
#nnz[filename] = OrderedDict((subj, len(nx.edges(graphs[filename][subj])))
#                        for subj in graphs[filename])
#write(outfile, 'nnz', nnz)

