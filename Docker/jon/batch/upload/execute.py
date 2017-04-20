#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import boto3
from argparse import ArgumentParser
import csv
import sys

sys.path.insert(0, '../')

if __name__ == "__main__":
    bucket = sys.argv[1]
    data = sys.argv[2] # the name of the text file with the arguments for (typically "arguments.txt")
    s3 = boto3.client('s3')
    s3.download_file(bucket, data, data)
    with open(data, "r") as args_list:
        args = args_list.read().split(' ')

    print('bucktet: %s' % args[0])
    print('data: %s' % args[1])

    # looping through all the html outputs in the 'output' directory, and pushing each of them to s3
    directory = 'output'
    for filename in os.listdir(directory):
        if filename.endswith(".html"):
            # print(os.path.join(directory, filename))
            file_path = os.path.join(directory, filename)
            key = filename
            s3.upload_file(file_path, bucket, key)
            continue
        else:
            continue

    # key = "testdataoutput.txt"
    # s3.upload_file("testdataoutput.txt", bucket, key)

