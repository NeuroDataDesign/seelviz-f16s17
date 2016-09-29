#!/usr/bin/env python
#-*- coding:utf-8 -*-
from __future__ import print_function

__author__ = 'seelviz'

import matplotlib as mpl
mpl.use('Agg')

from skimage import data, img_as_float
from skimage import exposure

import cv2

import math, os, gc
import numpy as np
import nibabel as nib

"""
clarity.py
"""

class clarityviz(object):
    """This class loads the csv, plots its Nodes and Edges, and extracts the data into an xml file"""

    def __init__(self, token, source_directory = None):
        """Constructor that takes in the token name, loads the img, and makes a directory."""
        self._token = token # Token
        self._img = None    # Image Data
        self._shape = None  # (x, y, z)
        self._max = None    # Max Value
        self._points = None
        self._source_directory = source_directory

        self._infile = None
        self._nodefile = None
        self._edgefile = None
        self._filename = None

        #self.loadImg(self._token + ".img")
        
        #self.loadEqImg()

        # make a directory if none exists
        if not os.path.exists(token):
            os.makedirs(token)

    def getShape(self):
        return self._shape

    def getMax(self):
        return self._max

    def discardImg(self):
        del self._img
        gc.collect()
        return self

    def generate_plotly_html(self):
        # Type in the path to your csv file here
        thedata = np.genfromtxt(self._token + '/' + self._token + '.csv',
            delimiter=',', dtype='int', usecols = (0,1,2), names=['a','b','c'])

        trace1 = go.Scatter3d(
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
        layout = go.Layout(
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=0
            )
        )

        fig = go.Figure(data=data, layout=layout)
        print(self._token + "plotly")
        plotly.offline.plot(fig, filename= self._token + "/" + self._token + "plotly.html")

    def generateHistogram(self):
        print('Generating Histogram...')
        if self._source_directory == None:
            path = self._token + '.img'
        else:
            path = self._source_directory + "/" + self._token + ".img"

        im = nib.load(path)

        im = im.get_data()
        img = im[:,:,:]

        shape = im.shape
        #affine = im.get_affine()

        x_value = shape[0]
        y_value = shape[1]
        z_value = shape[2]

        #####################################################

        imgflat = img.reshape(-1)

        #img_grey = np.array(imgflat * 255, dtype = np.uint8)

        #img_eq = exposure.equalize_hist(img_grey)

        #new_img = img_eq.reshape(x_value, y_value, z_value)
        #globaleq = nib.Nifti1Image(new_img, np.eye(4))

        #nb.save(globaleq, '/home/albert/Thumbo/AutAglobaleq.nii')

        ######################################################

        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        img_grey = np.array(imgflat * 255, dtype = np.uint8)
        #threshed = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)

        cl1 = clahe.apply(img_grey)

        #cv2.imwrite('clahe_2.jpg',cl1)
        #cv2.startWindowThread()
        #cv2.namedWindow("adaptive")
        #cv2.imshow("adaptive", cl1)
        #cv2.imshow("adaptive", threshed)
        #plt.imshow(threshed)

        localimgflat = cl1 #cl1.reshape(-1)

        newer_img = localimgflat.reshape(x_value, y_value, z_value)
        localeq = nib.Nifti1Image(newer_img, np.eye(4))
        nib.save(localeq, self._token + '/' + self._token + 'localeq.nii')

    def loadImg(self, path=None, info=False):
        """Method for loading the .img file"""
#        if path is None:
#            path = rs.RAW_DATA_PATH
#        pathname = path + self._token+".img"
        if self._source_directory == None:
            path = self._token + '.img'
        else:
            path = self._source_directory + "/" + self._token + ".img"
        
        #path = self._token + '.hdr'

        img = nib.load(path)
        if info:
            print(img)
        self._img = img.get_data()[:,:,:,0]
        self._shape = self._img.shape
        self._max = np.max(self._img)
        print("Image Loaded: %s"%(path))
        return self
    
    def loadEqImg(self, path=None, info=False):
        print('Inside loadEqImg')
        if self._source_directory == None:
            path = self._token + '.img'
        else:
            path = self._source_directory + "/" + self._token + ".img"
        print("Loading: %s"%(path))

        #pathname = path+self._token+".nii"
        img = nib.load(path)
        if info:
            print(img)
        self._img = img.get_data()
        self._shape = self._img.shape
        self._max = np.max(self._img)
        print("Image Loaded: %s"%(path))
        return self

    def loadGeneratedNii(self, path=None, info=False):
        path = self._token + '/' + self._token + 'localeq.nii'
        print("Loading: %s"%(path))

        #pathname = path+self._token+".nii"
        img = nib.load(path)
        if info:
            print(img)
        #self._img = img.get_data()[:,:,:,0]
        self._img = img.get_data()
        self._shape = self._img.shape
        self._max = np.max(self._img)
        print("Image Loaded: %s"%(path))
        return self


    def loadNii(self, path=None, info=False):
        if self._source_directory == None:
            path = self._token + '.nii'
        else:
            path = self._source_directory + "/" + self._token + ".nii"
        print("Loading: %s"%(path))

        #pathname = path+self._token+".nii"
        img = nib.load(path)
        if info:
            print(img)
        #self._img = img.get_data()[:,:,:,0]
        self._img = img.get_data()
        self._shape = self._img.shape
        self._max = np.max(self._img)
        print("Image Loaded: %s"%(path))
        return self

    def loadInitCsv(self, path=None):
        """Method for loading the initial csv file"""
        self._infile = open(path, 'r')
        self._filename = self._infile.name[:-4] if self._infile.name.endswith('.csv') else self._infile.name
        print("File Loaded: %s"%(self._infile.name))
        return self

    def loadNodeCsv(self, path=None):
        """Method for loading the nodes csv file"""
        self._nodefile = open(path, 'r')
        print("File Loaded: %s"%(self._nodefile.name))
        return self

    def loadEdgeCsv(self, path=None):
        """Method for loading the edges csv file"""
        self._edgefile = open(path, 'r')
        print("File Loaded: %s"%(self._edgefile.name))
        return self

    def imgToPoints(self, threshold=0.1, sample=0.5, optimize=True):
        """Method to extract points data from the img file."""
        if not 0 <= threshold < 1:
            raise ValueError("Threshold should be within [0,1).")
        if not 0 < sample <= 1:
            raise ValueError("Sample rate should be within (0,1].")
        if self._img is None:
            raise ValueError("Img haven't loaded, please call loadImg() first.")

        total = self._shape[0]*self._shape[1]*self._shape[2]
        print("Coverting to points...\ntoken=%s\ntotal=%d\nmax=%f\nthreshold=%f\nsample=%f"\
               %(self._token,total,self._max,threshold,sample))
        print("(This will take couple minutes)")
        # threshold
        filt = self._img > threshold * self._max
        x, y, z = np.where(filt)
        v = self._img[filt]
        if optimize:
            self.discardImg()
        v = np.int16(255*(np.float32(v)/np.float32(self._max)))
        l = v.shape
        print("Above threshold=%d"%(l))
        # sample
        if sample < 1.0:
            filt = np.random.random(size=l) < sample
            x = x[filt]
            y = y[filt]
            z = z[filt]
            v = v[filt]
        self._points = np.vstack([x,y,z,v])
        self._points = np.transpose(self._points)
        print("Samples=%d"%(self._points.shape[0]))
        print("Finished")
        return self

    def savePoints(self,path=None):
        if self._points is None:
            raise ValueError("Points is empty, please call imgToPoints() first.")

        pathname = self._token + "/" + self._token+".csv"
        np.savetxt(pathname,self._points,fmt='%d',delimiter=',')
        return self

    def plot3d(self, infile = None):
        """Method for plotting the Nodes and Edges"""
        filename = ""
        points_file = None
        if infile == None:
            points_file = self._points
            filename = self._token
        else: 
            self.loadInitCsv(infile)
            infile = self._infile
            filename = self._filename
        
        # points is an array of arrays
        points = self._points
        outpath = self._token + '/'
        nodename = outpath + filename + '.nodes.csv'
        edgename = outpath + filename + '.edges.csv'

#        for line in points_file:
#            line = line.strip().split(',')
#            points.append(str(line[0]) + "," + str(line[1]) + "," + str(line[2]))

        with open(nodename, 'w') as nodefile:
            with open(edgename, 'w') as edgefile:
                for ind in range(len(points)):
                    #temp = points[ind].strip().split(',')
                    temp = points[ind]
                    x = temp[0]
                    y = temp[1]
                    z = temp[2]
                    v = temp[3]
                    radius = 18
                    nodefile.write("s" + str(ind + 1) + "," + str(x) + "," + str(y) + "," + str(z) + "\n")
                    for index in range(ind + 1, len(points)):
                        tmp = points[index]
                        distance = math.sqrt(math.pow(int(x) - int(tmp[0]), 2) + math.pow(int(y) - int(tmp[1]), 2) + math.pow(int(z) - int(tmp[2]), 2))
                        if distance < radius:
                                edgefile.write("s" + str(ind + 1) + "," + "s" + str(index + 1) + "\n")
                self._nodefile = nodefile
                self._edgefile = edgefile
                    
    def graphmlconvert(self, nodefilename = None, edgefilename = None):
        """Method for extracting the data to a graphml file, based on the node and edge files"""
        nodefile = None
        edgefile = None

        # If no nodefilename was entered, used the Clarity object's nodefile
        if nodefilename == None: 
            #nodefile = self._nodefile
            #nodefile = open(self._nodefile, 'r')
            
            self.loadNodeCsv(self._token + "/" + self._token + ".nodes.csv")
            nodefile = self._nodefile
        else:
            self.loadNodeCsv(nodefilename)
            nodefile = self._nodefile
            
        # If no edgefilename was entered, used the Clarity object's edgefile
        if edgefilename == None: 
            #edgefile = self._edgefile
            #edgefile = open(self._edgefile, 'r')
            
            self.loadEdgeCsv(self._token + "/" + self._token + ".edges.csv")
            edgefile = self._edgefile
        else:
            self.loadEdgeCsv(edgefilename)
            edgefile = self._edgefile

        # Start writing to the output graphml file
        path = self._token + "/" + self._token + ".graphml"
        with open(path, 'w') as outfile:
            outfile.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
            outfile.write("<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\"\n")
            outfile.write("         xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n")
            outfile.write("         xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns\n")
            outfile.write("         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\">\n")

            outfile.write("  <key id=\"d0\" for=\"node\" attr.name=\"attr\" attr.type=\"string\"/>\n")
            outfile.write("  <key id=\"e_weight\" for=\"edge\" attr.name=\"weight\" attr.type=\"double\"/>\n")
            outfile.write("  <graph id=\"G\" edgedefault=\"undirected\">\n")

            for line in nodefile:
                 if len(line) == 0:
                     continue
                 line = line.strip().split(',')
                 outfile.write("    <node id=\"" + line[0] + "\">\n")
                 outfile.write("      <data key=\"d0\">[" + line[1] + ", " + line[2] + ", " + line[3] +"]</data>\n")
                 outfile.write("    </node>\n")

            for line in edgefile:
                 if len(line) == 0:
                     continue
                 line = line.strip().split(',')
                 outfile.write("    <edge source=\"" + line[0] + "\" target=\"" + line[1] + "\">\n")
                 outfile.write("      <data key=\"e_weight\">1</data>\n")
                 outfile.write("    </edge>\n")

            outfile.write("  </graph>\n</graphml>")

    def graphmlToPlotly(self, path):
        ## Type in the path to your csv file here
        thedata = np.genfromtxt('../data/points/localeq.csv', delimiter=',', dtype='int', usecols = (0,1,2), names=['a','b','c'])

        trace1 = go.Scatter3d(
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
        layout = go.Layout(
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=0
            )
        )
            
        fig = go.Figure(data=data, layout=layout)
        print("localeq")
        plotly.offline.plot(fig, filename= "localeq")



