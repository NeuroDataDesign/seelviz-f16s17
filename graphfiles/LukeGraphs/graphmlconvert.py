import sys

with open(sys.argv[1], 'r') as nodes:
    with open(sys.argv[2], 'r') as edges:
        with open(sys.argv[3], 'w') as outfile:
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
                 outfile.write("      <data key=\"e_weight\">1</data>\n")
                 outfile.write("    </edge>\n")

            outfile.write("  </graph>\n</graphml>")

