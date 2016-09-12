import re
import sys

brightest = []
data = {}
values = {}

with open(sys.argv[1], 'r') as infile:
# with open("C:\\Users\\L\\Documents\\Homework\\BME\\Neuro Data I\\tmp\\seelviz\\localeq.csv", "r") as infile:
    for line in infile:
        line = line.strip().split(',')
        values[str(line[3])] = values.get(str(line[3]), 0) + 1
        if len(data) == 100000:
            tmpkey = min(data, key=data.get)
            if line[3] > data[tmpkey]:
                data.pop(tmpkey, None)
                data[str(line[0] + "," + line[1] + "," + line[2])] = line[3]
            else:
                continue
        elif len(data) < 100000:
            data[str(line[0] + "," + line[1] + "," + line[2])] = line[3]

    with open(sys.argv[2], 'w') as outfile:
    # with open("C:\\Users\\L\\Documents\\Homework\\BME\\Neuro Data I\\tmp\\seelviz\\out.csv", "w") as outfile:
        for key in data:
            outfile.write(key + "," + str(data[key]) + "\n")

for key in values:
    print(str(key) + "\t" + str(values[key]) + "\n")
