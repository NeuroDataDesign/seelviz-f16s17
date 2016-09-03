import clearity as cl  # I wrote this module for easier operations on data
import clearity.resources as rs
import csv,gc  # garbage memory collection :)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import jgraph as ig

####Normal Image

c = cl.Clarity('Fear197')
    #fname = rs.HIST_DATA_PATH+token+".csv"
claritycsv = c.loadNiiImg().imgToPoints(threshold=0.1,sample=0.11).savePoints()
    #np.savetxt(token,claritycsv,delimiter=',')
print "Fear197.csv saved."
del c
gc.collect()

####General Image

c = cl.Clarity('globaleq')
    #fname = rs.HIST_DATA_PATH+token+".csv"
claritycsv = c.loadEqImg().imgToPoints(threshold=0.99,sample=0.005).savePoints()
    #np.savetxt(token,claritycsv,delimiter=',')
print "globaleq.csv saved."
del c
gc.collect()

####Local Image

c = cl.Clarity('localeq')
    #fname = rs.HIST_DATA_PATH+token+".csv"
claritycsv = c.loadEqImg().imgToPoints(threshold=0.99,sample=0.03).savePoints()
    #np.savetxt(token,claritycsv,delimiter=',')
print "localeq.csv saved."
del c
gc.collect()