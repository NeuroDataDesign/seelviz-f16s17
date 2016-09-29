import re
import sys
import random

random.seed(9001)

#names = ['Cocaine174localeq','Cocaine175localeq','Cocaine178localeq','Control181localeq','Control182localeq','Control189localeq','Control239localeq','Control258localeq','Fear187localeq','Fear197localeq','Fear200localeq']
names = ['Aut1360localeq','Aut1367localeq','Aut1374localeq','AutAlocaleq']

#values = {}

#with open(sys.argv[1], 'r') as infile:
# with open("C:\\Users\\L\\Documents\\Homework\\BME\\Neuro Data I\\tmp\\seelviz\\localeq.csv", "r") as infile:
for name in names:
    fin = 'csv/' + name + '.csv'
    fout = 'csv/' + name + '.5000.brightest.csv'
    brightest = []
    data = {}
    bright255 = []
    bright254 = []
    bright253 = []

    with open(fin, 'r') as infile:
        index255 = []
        index254 = []
        index253 = []
        #with open(sys.argv[2], 'w') as outfile:
        with open(fout, 'w') as outfile:
            for line in infile:
                line = line.strip().split(',')
                if int(line[3]) == 255:
                    bright255.append(str(line[0] + "," + line[1] + "," + line[2]))
                elif int(line[3]) == 254:
                    bright254.append(str(line[0] + "," + line[1] + "," + line[2]))
                elif int(line[3]) == 253:
                    bright253.append(str(line[0] + "," + line[1] + "," + line[2]))
             
            len255 = len(bright255)
            len254 = len(bright254)
            len253 = len(bright253)
            if len255 > 5000:
                index255 = random.sample(xrange(0, len255), 5000)
                for ind in index255:
                    outfile.write(str(bright255[ind]) + ",255\n")
            else:
                for item in bright255:
                    outfile.write(str(item) + ",255\n")
                if (len255 + len254) > 5000:
                    index254 = random.sample(xrange(0, len254), 5000-len255)
                    for ind in index254:
                        outfile.write(str(bright254[ind]) + ",254\n")
                #else:
                #    for item in bright254:
                #        outfile.write(str(item) + ",254\n")
                #    index253 = random.sample(xrange(0, len253), (5000-len255-len254))
                #    for ind in index253:
                 #        outfile.write(str(bright253[ind]) + ",253\n")
     
