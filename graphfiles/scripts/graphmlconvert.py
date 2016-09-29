import sys

files = ['Cocaine174localeq.5000','Cocaine175localeq.5000','Cocaine178localeq.5000','Control181localeq.5000','Control182localeq.5000','Control189localeq.5000','Control239localeq.5000','Control258localeq.5000','Fear187localeq.5000','Fear197localeq.5000','Fear199localeq.5000','Fear200localeq.5000','Aut1360localeq.5000','Aut1374localeq.5000']
#files = ['Fear199localeq.5000']
for f in files:
    fnode = 'graphml/edges100k/' + f + '.brightest.nodes.csv'
    fedges = 'graphml/edges100k/' +  f + '.brightest.edges.csv'
    fout = 'Fear199graphml/' + 'subsample/' + f + '.100k.graphml'
    #with open(sys.argv[1], 'r') as nodes:
    with open(fnode, 'r') as nodes:
        #with open(sys.argv[2], 'r') as edges:
        with open(fedges, 'r') as edges:
            #with open(sys.argv[3], 'w') as outfile:
            with open(fout, 'w') as outfile:
                outfile.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
                outfile.write("<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\"\n")
                outfile.write("         xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n")
                outfile.write("         xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns\n")
                outfile.write("         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\">\n")

                outfile.write("  <key id=\"d0\" for=\"node\" attr.name=\"attr\" attr.type=\"string\"/>\n")
                outfile.write("  <key id=\"e_weight\" for=\"edge\" attr.name=\"weight\" attr.type=\"double\"/>\n")
                outfile.write("  <graph id=\"G\" edgedefault=\"undirected\">\n")

                for line in nodes:
                    if len(line) == 0:
                        continue
                    line = line.strip().split(',')
                    outfile.write("    <node id=\"" + line[0] + "\">\n")
                    outfile.write("      <data key=\"d0\">[" + line[1] + ", " + line[2] + ", " + line[3] +"]</data>\n")
                    outfile.write("    </node>\n")

                for line in edges:
                    if len(line) == 0:
                        continue
                    line = line.strip().split(',')
                    outfile.write("    <edge source=\"" + line[0] + "\" target=\"" + line[1] + "\">\n")
                    outfile.write("      <data key=\"e_weight\">" + "1" + "</data>\n")
                    outfile.write("    </edge>\n")

                outfile.write("  </graph>\n</graphml>")

