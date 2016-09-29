#!/usr/bin/env python

import sys
import math

points = []
#files = ['Control182localeq','Control189localeq','Control239localeq']

class Plot3d():

	def getNodesAndEdges(infile):
		inpath = 'csv/'
		outpath = 'graphml/'
		filename = infile.name[:-4} if infile.name.endswith('.csv') else infile.name
		infilename = inpath + sys.argv[1] + '.csv'
		nodename = outpath + sys.argv[1] + '.nodes.csv'
		edgename = outpath + sys.argv[1] + '.edges.csv'

		
		for line in infile:
		line = line.strip().split(',')
		points.append(str(line[0]) + "," + str(line[1]) + "," + str(line[2]))         

		#with open(sys.argv[2], 'w') as nodefile:
		with open(nodename, 'w') as nodefile:
			#with open(sys.argv[3], 'w') as edgefile:
			with open(edgename, 'w') as edgefile:
			for ind in range(len(points)):
				temp = points[ind].strip().split(',')
				x = temp[0]
				y = temp[1]
				z = temp[2]
				radius = 18
				nodefile.write("s" + str(ind + 1) + "," + str(x) + "," + str(y) + "," + str(z) + "\n")
				for index in range(ind + 1, len(points)):
				tmp = points[index].strip().split(',')
				distance = math.sqrt(math.pow(int(x) - int(tmp[0]), 2) + math.pow(int(y) - int(tmp[1]), 2) + math.pow(int(z) - int(tmp[2]), 2))
				if distance < radius:
					edgefile.write("s" + str(ind + 1) + "," + "s" + str(index + 1) + "\n")

		return [nodefile, edgefile]

#old shit =======================================================================
#for f in files:
#    filename = inpath + f + '.csv'
#    nodename = outpath + f + '.nodes.csv'
#    edgename = outpath + f + '.edges.csv'
#    #with open(sys.argv[1], 'r') as infile:
#    with open(filename, 'r') as infile:
#        for line in infile:
#            line = line.strip().split(',')
#            points.append(str(line[0]) + "," + str(line[1]) + "," + str(line[2]))         
#    
#    #with open(sys.argv[2], 'w') as outfile:
#    with open(nodename, 'w') as outfile:
#        #with open(sys.argv[3], 'w') as edgefile:
#        with open(edgename, 'w') as edgefile:
#            for ind in range(len(points)):
#                temp = points[ind].strip().split(',')
#                x = temp[0]
#                y = temp[1]
#                z = temp[2]
#                radius = 18
#                outfile.write("s" + str(ind + 1) + "," + str(x) + "," + str(y) + "," + str(z) + "\n")
#                for index in range(ind + 1, len(points)):
#                    tmp = points[index].strip().split(',')
#                    distance = math.sqrt(math.pow(int(x) - int(tmp[0]), 2) + math.pow(int(y) - int(tmp[1]), 2) + math.pow(int(z) - int(tmp[2]), 2))
#                    if distance < radius:
#                        edgefile.write("s" + str(ind + 1) + "," + "s" + str(index + 1) + "\n")

