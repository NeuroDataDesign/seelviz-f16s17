#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import boto3
from argparse import ArgumentParser
import csv
import sys

import clarityviz as clv

sys.path.insert(0, '../')

def run_pipeline(token, resolution=5, points_path='', regis_path=''):
    """
    Runs each individual part of the pipeline.
    :param token: Token name for REGISTERED brains in NeuroData.  (https://dev.neurodata.io/nd/ca/public_tokens/)
    :param orientation: Orientation of brain in NeuroData (eg: LSA, RSI)
    :param resolution: Resolution for the brains in NeuroData (from raw image data at resolution 0 to ds levels at resolution 5)
    :param points_path: The path to the csv of points, skipping downloading, registration, and clahe
    :param regis_path: The path to the nii of the registered brain, skipping downloading.
    """
    path = ''
    if points_path == '':
        if regis_path == '':
            clv.analysis.get_registered(token)
            path = "img/" + token + "_regis.nii"  #why _anno?  That's the refAnnoImg...
        else:
            path = regis_path
        im = clv.analysis.apply_clahe(path);
        output_ds = clv.analysis.downsample(im, num_points=10000);
        clv.analysis.save_points(output_ds, "points/" + token + ".csv")
        points_path = "points/" + token + ".csv";
    clv.analysis.generate_pointcloud(points_path, "output/" + token + "_pointcloud.html");
    # clv.analysis.get_atlas_annotate(save=True);
    clv.analysis.get_regions(points_path, "atlas/ara3_annotation.nii", "points/" + token + "_regions.csv");
    points_region_path = "points/" + token + "_regions.csv";
    g = clv.analysis.create_graph(points_region_path, output_filename="graphml/" + token + "_graph.graphml");
    clv.analysis.plot_graphml3d(g, output_path="output/" + token + "_edgegraph.html");
    clv.analysis.generate_region_graph(token, points_region_path, output_path="output/" + token + "_regions.html");
    clv.analysis.generate_density_graph(graph_path="graphml/" + token + "_graph.graphml", output_path="output/" + token + "_density.html", plot_title="False-Color Density of " + token);
    print("Completed pipeline...!")

if __name__ == "__main__":
    bucket = sys.argv[1]
    token = sys.argv[2] # token, if token == 'test' then it runs a test pipeline
    s3 = boto3.client('s3')

    # pushing indicator file to s3
    f = open('file.txt','w')
    message = 'About to start pipeline for token: ' + token
    f.write(message)
    f.close()
    key = 'file.txt'
    s3.upload_file('file.txt', bucket, key)

    if token != 'test':
        run_pipeline(token, 5)
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
