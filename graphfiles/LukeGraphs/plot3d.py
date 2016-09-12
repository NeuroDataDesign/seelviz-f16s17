import sys
import math

points = []
with open(sys.argv[1], 'r') as infile:
    for line in infile:
        line = line.strip().split(',')
        points.append(str(line[0]) + "," + str(line[1]) + "," + str(line[2]))         
    
with open(sys.argv[2], 'w') as outfile:
    with open(sys.argv[3], 'w') as edgefile:
        for ind in range(len(points)):
            temp = points[ind].strip().split(',')
            x = temp[0]
            y = temp[1]
            z = temp[2]
            radius = 15
            outfile.write("s" + str(ind + 1) + "," + str(x) + "," + str(y) + "," + str(z) + "\n")
            for index in range(ind + 1, len(points)):
                tmp = points[index].strip().split(',')
                distance = math.sqrt(math.pow(int(x) - int(tmp[0]), 2) + math.pow(int(y) - int(tmp[1]), 2) + math.pow(int(z) - int(tmp[2]), 2))
                if distance < radius:
                    edgefile.write("s" + str(ind + 1) + "," + "s" + str(index + 1) + "\n")

