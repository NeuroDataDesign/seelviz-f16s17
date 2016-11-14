# from django.views import generic
# from .models import TokenUpload

# class IndexView(generic.ListView):
#   template_name = 'clarityviz/index.html'

#   def get_queryset(self):
#       return 


from django.shortcuts import render
from django.template.loader import render_to_string
from django.conf import settings
# from django.template import loader
from django.http import HttpResponse
from .models import TokenUpload

# non django stuff ========

import os
import os.path

import shutil
import tempfile
import glob
import random

from clarityviz.templates.clarityviz import claritybase
from clarityviz.templates.clarityviz import densitygraph
from clarityviz.templates.clarityviz import atlasregiongraph
import networkx as nx
import plotly

from ndreg import *
import ndio.remote.neurodata as neurodata
import nibabel as nb
from numpy import genfromtxt

import time

from clarityviz.templates import test

def index(request):
    # return HttpResponse("<h2>Hello World</h2>")

    # get token from user form, then pass that token into the url 
    # while running the pipeline, then open the link to the output 
    # with the token e.g. /clarityviz/fear199

    # template = loader.get_template('clarityviz/index.html')

    # return HttpResponse(template.render(context, request))

    # uploads = TokenUpload.objects.all()
    # context = {'uploads': uploads}

    # if(request.GET.get('tokenButton')):
    #     test.testFunction( request.GET.get('myToken') )
    return render(request, 'clarityviz/index.html')
    # return render(request, 'clarityviz/index.html', context)


def token_compute(request):
    print('INSIDE TOKEN_COMPUTE')
    token = request.POST['token']
    ogToken = token
    if token != 'Aut1367reorient_atlas':
        token = token.strip().split(',')
        ori1 = token[1].strip()
        token = token[0].strip()

    # test.testFunction(token)

    if token != 'Aut1367reorient_atlas':
        ip_start = time.time()
        token = image_parse(token,ori1)
        ip_run_time = time.time() - ip_start
        print('image_parse total time = %f' % ip_run_time)

        start = time.time()
        density_graph(token)
        run_time = time.time() - start
        print('density_graph total time = %f' % run_time)
        
        start = time.time()
        atlas_region(token)
        run_time = time.time() - start
        print('density_graph total time = %f' % run_time)
    
    fzip = shutil.make_archive(token, 'zip', token)
    fzip_abs = os.path.abspath(fzip)

    html = """
    {% extends "clarityviz/header.html" %}

    {% block content %}

    <header>
        <div class="header-content">
            <div class="header-content-inner">
                {% if token %}
                    <h1>{{token}}</h1>
                {% endif %}
            </div>
        </div>
    </header>

    <body>

    <section class="bg-graph" id="about">
        <div class="container">
            <div class="row">
                <div class="col-lg-8 col-lg-offset-2 text-center">
                    <h2 class="section-heading">Results</h2>
                    <hr class="light">
    """

    plotly = []
    file_paths = []
    file_basenames = []
    plotly_paths = []
    plotly_basenames = []
    # for filename in glob.glob(token + '/*'):
    #     absPath = os.path.abspath(filename)
    #     if os.path.isdir(absPath):
    #         link = '<a href="/index?directory=' + absPath + '">' + os.path.basename(filename) + "</a> <br />"
    #         html += link
    #     else:
    #         if filename.endswith('html'):
    #             plotly.append(filename)
    #         link = '<a href="/download/?filepath=' + absPath + '">' + os.path.basename(filename) + "</a> <br />"
    #         html += link

    # for plot in plotly:
    #     absPath = os.path.abspath(plot)
    #     html += """
    #       <form action="plotly" method="get">
    #         <input type="text" value=""" + '"' + absPath + '" name="plot" ' + """/>
    #         <button type="submit">View """ + os.path.basename(plot) + """</button>
    #       </form>"""
    #     # html += '<a href="file:///' + '//' + absPath + '">' + "View Plotly graph</a> <br />"    

    # for plot in plotly:
    #     absPath = os.path.abspath(plot)
    #     html += """
    #       <form action="plotly" method="get">
    #         <input type="text" value=""" + '"' + absPath + '" name="plot" ' + """/>
    #         <button type="submit">View """ + os.path.basename(plot) + """</button>
    #       </form>"""
    #     # html += '<a href="file:///' + '//' + absPath + '">' + "View Plotly graph</a> <br />"

    for filename in glob.glob('output/' + token + '/*'):
        absPath = os.path.abspath(filename)
        if os.path.isdir(absPath):
            link = '<a href="/clarityviz/download/' + absPath + '">' + os.path.basename(filename) + "</a> <br />"
            html += link
        else:
            if filename.endswith('html'):
                plotly.append(filename)
            link = '<a href="/clarityviz/download/' + absPath + '">' + os.path.basename(filename) + "</a> <br />"
            html += link

    html += '<a href="/clarityviz/download/' + fzip_abs + '">' + token + '.zip' + "</a> <br /><br />"

    for plot in plotly:
        absPath = os.path.abspath(plot)
        if absPath.endswith('_brain_pointcloud.html'):
            link = '<a href="/clarityviz/plot/' + absPath + '" class="page-scroll btn btn-default btn-xl sr-button">Brain Pointcloud</a> <br /><br />'
        elif absPath.endswith('_edge_count_pointcloud.html'):
            link = '<a href="/clarityviz/plot/' + absPath + '" class="page-scroll btn btn-default btn-xl sr-button">Edge Count Pointcloud</a> <br /><br />'
        elif absPath.endswith('_density_pointcloud.html'):
            link = '<a href="/clarityviz/plot/' + absPath + '" class="page-scroll btn btn-default btn-xl sr-button">Density Pointcloud</a> <br /><br />'
        elif absPath.endswith('_density_pointcloud_heatmap.html'):
            link = '<a href="/clarityviz/plot/' + absPath + '" class="page-scroll btn btn-default btn-xl sr-button">Density Pointcloud Heatmap</a> <br /><br />'
        elif absPath.endswith('_region_pointcloud.html'):
            link = '<a href="/clarityviz/plot/' + absPath + '" class="page-scroll btn btn-default btn-xl sr-button">Atlas Region Pointcloud</a> <br /><br />'
        html += link

    # for plot in plotly:
    #     absPath = os.path.abspath(plot)
    #     with open(absPath, "r") as ins:
    #         for line in ins:
    #             html += line


    html += """
                </div>
            </div>
        </div>
    </section>
    </body>
    </html>
    {% endblock %}
    """

    with open("clarityviz/templates/clarityviz/files.html", "w+") as text_file:
        text_file.write("{}".format(html))

    # files = glob.glob(token + '/*')

    context = {'token': token}

    # return HttpResponse(html)

    return render(request, 'clarityviz/files.html', context) 

def download(request, path):
    file_path = path
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/x-download")
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
            return response
    else:
        raise Http404

def plot(request, path):

    html = """
    {% extends "clarityviz/header.html" %}

    {% block content %}

    <header>
        <div class="header-content">
            <div class="header-content-inner">
                {% if type %}
                    <h1>{{type}}</h2>
                {% endif %}
            </div>
        </div>
    </header>

    <body>

    <section class="bg-graph" id="about">
        <div class="container">
            <div class="row">
                <div class="col-lg-8 col-lg-offset-2 text-center">
    """

    with open(path, "r") as ins:
        for line in ins:
            html += line

    html += """
                </div>
            </div>
        </div>
        <div class="container">
            {% if description %}
                <p>{{description}}</p>
            {% endif %}
        </div>
    </section>
    </body>
    </html>
    {% endblock %}
    """

    with open("clarityviz/templates/clarityviz/plot.html", "w+") as text_file:
        text_file.write("{}".format(html))

    plot_type = ''
    description = ''
    if path.endswith('_brain_pointcloud.html'):
        plot_type = 'Brain Pointcloud'
        description = 'In the plot above we have a point cloud visualization of the 10,000 brightest points of the CLARITY brain selected after image filtering and histogram equalization.  The filtering and histogram equalization increased the relative contrast of each voxel relative to its nearest neighbors; the 10,000 brightest points were selected by randomly sampling voxels with 255 grey scale values.  We hypothesize that the denser areas of the point cloud correspond to brain regions with the more neurological activity.'
    elif path.endswith('_edge_count_pointcloud.html'):
        plot_type = 'Edge Count Pointcloud'
        description = '''This purple node and cyan edge plot shows the connections from the density plot.  Each cyan edge was drawn with the same epsilon ball initialization used for the density plot.  It's important to note that the process of finding all the edges for a given node is a significant computational task that scales exponentially with increased epsilon ball radius.  The most connected nodes may show some properties of interest'''
    elif path.endswith('_density_pointcloud.html'):
        plot_type = 'Density Pointcloud'
        description = 'The multicolored plot shows a false-coloration scheme of the 10,000 brightest points by their edge counts, relative to a preselected epsilon ball radius.  The epsilon ball radius determines the number of edges a given node has by connecting all neighboring nodes within the radius with an edge.  Black nodes had an edge count of 0.  Then, in reverse rainbow order, (purple to red), we get increasing numbers of edges.  The densest node with the most edges is shown in white.  The plot supports up to 20 different colors.'
    elif path.endswith('_density_pointcloud_heatmap.html'):
        plot_type = 'Density Pointcloud Heatmap'
    elif path.endswith('_region_pointcloud.html'):
        plot_type = 'Atlas Region Pointcloud'
        description = 'This graph shows a plot of the brain with each region as designated by the atlas a unique colored. Controls along the side allow for toggling the traces on/off'

    context = {'type': plot_type, 'description': description}

    return render(request, 'clarityviz/plot.html', context) 


def output(request, token):
    return render(request, 'clarityviz/output.html')


def imgGet(inToken, ori1):
    refToken = "ara_ccf2"                         # hardcoded 'ara_ccf2' atlas until additional functionality is requested
    refImg = imgDownload(refToken)                # download atlas
    refAnnoImg = imgDownload(refToken, channel="annotation")
    print "reference token/atlas obtained"
    inImg = imgDownload(inToken, resolution=5)    # store downsampled level 5 brain to memory
    (values, bins) = np.histogram(sitk.GetArrayFromImage(inImg), bins=100, range=(0,500))
    print "level 5 brain obtained"
    counts = np.bincount(values)
    maximum = np.argmax(counts)

    lowerThreshold = maximum
    upperThreshold = sitk.GetArrayFromImage(inImg).max()+1

    inImg = sitk.Threshold(inImg,lowerThreshold,upperThreshold,lowerThreshold) - lowerThreshold
    print "applied filtering"
    rawImg = sitk.GetArrayFromImage(inImg)
    xdimensions = len(rawImg[:,0,0])
    ydimensions = len(rawImg[0,:,0])
    zdimensions = len(rawImg[0,0,:])
    xyz = []
    for i in range(40000):
        value = 0
        while(value == 0):
            xval = random.sample(xrange(0,xdimensions), 1)[0]
            yval = random.sample(xrange(0,ydimensions), 1)[0]
            zval = random.sample(xrange(0,zdimensions), 1)[0]
            value = rawImg[xval,yval,zval]
            if [xval, yval, zval] not in xyz and value > 300:
                xyz.append([xval, yval, zval])
            else:
                value = 0
    print('inToken asdfasdf:')
    print(inToken)
    rImg = claritybase(inToken + 'raw', None)
    rImg.savePoints(None,xyz)
    rImg.generate_plotly_html()
    print "random sample of points above 250"
    spacingImg = inImg.GetSpacing()
    spacing = tuple(i * 50 for i in spacingImg)
    inImg.SetSpacing(spacingImg)
    inImg_download = inImg    # Aut1367 set to default spacing
    inImg = imgResample(inImg, spacing=refImg.GetSpacing())
    print "resampled img"
    Img_reorient = imgReorient(inImg, ori1, "RSA")    # reoriented Aut1367
    # Img_reorient = imgReorient(inImg, "LPS", "RSA")    # reoriented Aut1367
    refImg_ds = imgResample(refImg, spacing=spacing)    # atlas with downsampled spacing 10x
    inImg_ds = imgResample(Img_reorient, spacing=spacing)    # Aut1367 with downsampled spacing 10x
    print "reoriented image"
    affine = imgAffineComposite(inImg_ds, refImg_ds, iterations=100, useMI=True, verbose=True)
    inImg_affine = imgApplyAffine(Img_reorient, affine, size=refImg.GetSize())
    print "affine"
    inImg_ds = imgResample(inImg_affine, spacing=spacing)
    (field, invField) = imgMetamorphosisComposite(inImg_ds, refImg_ds, alphaList=[0.05, 0.02, 0.01], useMI=True, iterations=100, verbose=True)
    inImg_lddmm = imgApplyField(inImg_affine, field, size=refImg.GetSize())
    print "downsampled image"
    invAffine = affineInverse(affine)
    invAffineField = affineToField(invAffine, refImg.GetSize(), refImg.GetSpacing())
    invField = fieldApplyField(invAffineField, invField)
    inAnnoImg = imgApplyField(refAnnoImg, invField,useNearest=True, size=Img_reorient.GetSize())

    inAnnoImg = imgReorient(inAnnoImg, "RSA", ori1)
    # inAnnoImg = imgReorient(inAnnoImg, "RSA", "LPS")
    inAnnoImg = imgResample(inAnnoImg, spacing=inImg_download.GetSpacing(), size=inImg_download.GetSize(), useNearest=True)
    print "inverse affine"
    imgName = inToken + "reorient_atlas"
    location = "img/" + imgName + ".nii"
    imgWrite(inAnnoImg, str(location))
    # ndImg = sitk.GetArrayFromImage(inAnnoImg)
    # sitk.WriteImage(inAnnoImg, location)
    print "generated output"
    print imgName
    return imgName


def image_parse(inToken, ori1):
    start = time.time()
    # imgGet is where the token name changes to adding the 'reorient_atlas'
    imgName = imgGet(inToken,ori1)
    # imgName = imgGet(inToken)
    run_time = time.time() - start
    print('imgGet time = %f' % run_time)

    
    # imgName = inToken + 'reorient_atlas'
    copydir = os.path.join(os.getcwd(), os.path.dirname('img/'))
    print('copydir: %s' % copydir)
    print('imgName: %s' % imgName)
    img = claritybase(imgName, copydir)       # initial call for clarityviz
    print "loaded into claritybase"
    img.loadEqImg()
    print "loaded image"
    img.applyLocalEq()
    print "local histogram equalization"
    img.loadGeneratedNii()
    print "loaded generated nii"

    start = time.time()
    thr = 0.9
    sam = 0.0005
    # img.calculatePoints(threshold = thr, sample = sam)
    num_points = 10000
    img.calculatePointsByNumber(num_points)
    print "calculated points"
    run_time = time.time() - start
    print('calculatePoints time (with threshold = %f, sample = %f) = %f' % (thr, sam, run_time))

    # uncomment either (1) or (2)

    #(1)
    # start = time.time()
    # img.brightPoints(None,40000)
    # run_time = time.time() - start
    # print('brightPoints time = %f' % run_time)

    #(2)
    print "saving brightest points to csv"
    img.savePoints()

    print "generating plotly"
    start = time.time()
    img.generate_plotly_html()
    run_time = time.time() - start
    print('brightPoints time = %f' % run_time)

    print "generating nodes and edges list"
    start = time.time()
    img.plot3d()
    run_time = time.time() - start
    print('plot3d time = %f' % run_time)
    
    print "generating graphml"
    start = time.time()
    img.graphmlconvert()
    run_time = time.time() - start
    print('graphmlconvert time = %f' % run_time)

    print "generating density graph"
    img.get_brain_figure(None, imgName + ' edgecount')
    
    return imgName

def density_graph(Token):
    densg = densitygraph(Token)
    print 'densitygraph module'
    densg.generate_density_graph()
    print 'generated density graph'
    g = nx.read_graphml('output/' + Token + '/' + Token + '.graphml')
    ggraph = densg.get_brain_figure(g = g, plot_title=Token)
    plotly.offline.plot(ggraph, filename = 'output/' + Token + '/' + Token + '_density_pointcloud.html')
    hm = densg.generate_heat_map()
    plotly.offline.plot(hm, filename = 'output/' + Token + '/' + Token + '_density_pointcloud_heatmap.html')

def atlas_region(Token):
    atlas_img = 'output/' + Token + '/' + Token + 'localeq' + '.nii'
    atlas = nb.load(atlas_img)  # <- atlas .nii image
    atlas_data = atlas.get_data()

    csvfile = 'output/' + Token + '/' + Token + 'localeq.csv' # 'atlasexp/Control258localeq.csv'    # <- regular csv from the .nii to csv step

    bright_points = genfromtxt(csvfile, delimiter=',')

    locations = bright_points[:, 0:3]

    regions = [atlas_data[l[0], l[1], l[2]] for l in locations]

    outfile = open('output/' + Token + '/' + Token + '.region.csv', 'w')
    infile = open(csvfile, 'r')
    for i, line in enumerate(infile):
        line = line.strip().split(',')
        outfile.write(",".join(line) + "," + str(regions[i]) + "\n")    # adding a 5th column to the original csv indicating its region (integer)
    infile.close()
    outfile.close()

    print len(regions)
    print regions[0:10]
    uniq = list(set(regions))
    numRegions = len(uniq)
    print len(uniq)
    print uniq

    # newToken = Token + '.region'
    atlas = atlasregiongraph(Token)
    
    atlas.generate_atlas_region_graph(None, numRegions)
