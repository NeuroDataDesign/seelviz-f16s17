#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import boto3
from argparse import ArgumentParser
import csv
import sys

import clarityviz as clv

sys.path.insert(0, '../')

if __name__ == "__main__":
    bucket = sys.argv[1]
    token = sys.argv[2] # token, if token == 'test' then it runs a test pipeline
    s3 = boto3.client('s3')

    # pushing indicator file to s3
    f = open('file.txt','w')
    f.write('About to start pipeline')
    f.close()
    key = 'file.txt'
    s3.upload_file('file.txt', bucket, key)

    if token != 'test':
        clv.analysis.run_pipeline(args[0], int(args[1]))
    else:
        # Run test pipeline that generates dummy html's and to be uploaded
        html_str = """
        <table border=1>
             <tr>
               <th>Number</th>
               <th>Square</th>
             </tr>
             <indent>
             <% for i in range(10): %>
               <tr>
                 <td><%= i %></td>
                 <td><%= i**2 %></td>
               </tr>
             </indent>
        </table>
        """
        for i in range(5):
            file_name = 'htmlfile' + str(i) + '.html'
            Html_file = open(file_name, "w")
            Html_file.write(html_str)
            Html_file.close()

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
