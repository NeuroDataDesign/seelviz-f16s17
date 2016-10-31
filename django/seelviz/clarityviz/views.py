# from django.views import generic
# from .models import TokenUpload

# class IndexView(generic.ListView):
#   template_name = 'clarityviz/index.html'

#   def get_queryset(self):
#       return 


from django.shortcuts import render
# from django.template import loader
from django.http import HttpResponse
from .models import TokenUpload

import os
import os.path

import shutil
import tempfile
import glob
import random

from clarityviz import claritybase
from clarityviz import densitygraph
from clarityviz import atlasregiongraph
import networkx as nx
import plotly

from ndreg import *
import ndio.remote.neurodata as neurodata
import nibabel as nb
from numpy import genfromtxt

from . import test

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

    # test.testFunction(token)

    token = image_parse(token)
    density_graph(token)
    atlas_region(token)
    fzip = shutil.make_archive(token, 'zip', token)
    fzip_abs = os.path.abspath(fzip)

    plotly = []
    file_paths = []
    file_basenames = []
    plotly_paths = []
    plotly_basenames = []
    for filename in glob.glob(token + '/*'):
        absPath = os.path.abspath(filename)
        if os.path.isdir(absPath):
            link = '<a href="/index?directory=' + absPath + '">' + os.path.basename(filename) + "</a> <br />"
            html += link
        else:
            if filename.endswith('html'):
                plotly.append(filename)
            link = '<a href="/download/?filepath=' + absPath + '">' + os.path.basename(filename) + "</a> <br />"
            html += link

    for plot in plotly:
        absPath = os.path.abspath(plot)
        html += """
          <form action="plotly" method="get">
            <input type="text" value=""" + '"' + absPath + '" name="plot" ' + """/>
            <button type="submit">View """ + os.path.basename(plot) + """</button>
          </form>"""
        # html += '<a href="file:///' + '//' + absPath + '">' + "View Plotly graph</a> <br />"

    html += '<a href="/download/?filepath=' + fzip_abs + '">' + token + '.zip' + "</a> <br />"
    html += """</body></html>"""

    files = glob.glob(token + '/*')

    context = {'token': token, 'files': files}

    return HttpResponse(html)

    # return render(request, 'clarityviz/outputs.html', context) 

# def token_

def output(request, token):
    return render(request, 'clarityviz/outputs.html')

localDir = os.path.dirname(__file__)
absDir = os.path.join(os.getcwd(), localDir)
# print absDir

def imgGet(inToken):
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
    rImg = claritybase(inToken + 'raw', None)
    rImg.savePoints(None,xyz)
    rImg.generate_plotly_html()
    print "random sample of points above 250"
    spacingImg = inImg.GetSpacing()
    spacing = tuple(i * 10 for i in spacingImg)
    inImg.SetSpacing(spacingImg)
    inImg_download = inImg    # Aut1367 set to default spacing
    inImg = imgResample(inImg, spacing=refImg.GetSpacing())
    print "resampled img"
    Img_reorient = imgReorient(inImg, "LPS", "RSA")    # reoriented Aut1367
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

    inAnnoImg = imgReorient(inAnnoImg, "RSA", "LPS")
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


def image_parse(inToken):
    imgName = imgGet(inToken)
    # imgName = inToken + 'reorient_atlas'
    copydir = os.path.join(os.getcwd(), os.path.dirname('img/'))
    img = claritybase(imgName, copydir)       # initial call for clarityviz
    print "loaded into claritybase"
    img.loadEqImg()
    print "loaded image"
    img.applyLocalEq()
    print "local histogram equalization"
    img.loadGeneratedNii()
    print "loaded generated nii"
    img.calculatePoints(threshold = 0.9, sample = 0.05)
    print "calculating points"
    img.brightPoints(None,40000)
    print "saving brightest points to csv"
    # img.savePoints()
    img.generate_plotly_html()
    print "generating plotly"
    img.plot3d()
    print "generating nodes and edges list"
    img.graphmlconvert()
    print "generating graphml"
    img.get_brain_figure(None, imgName + ' edgecount')
    print "generating density graph"
    
    return imgName

def density_graph(Token):
    densg = densitygraph(Token)
    print 'densitygraph module'
    densg.generate_density_graph()
    print 'generated density graph'
    g = nx.read_graphml(Token + '/' + Token + '.graphml')
    ggraph = densg.get_brain_figure(g = g, plot_title=Token)
    plotly.offline.plot(ggraph, filename = Token + '/' + Token + '_brain_figure.html')
    hm = densg.generate_heat_map()
    plotly.offline.plot(hm, filename = Token + '/' + Token + '_brain_heatmap.html')

def atlas_region(Token):
    atlas_img = Token + '/' + Token + 'localeq' + '.nii'
    atlas = nb.load(atlas_img)  # <- atlas .nii image
    atlas_data = atlas.get_data()

    csvfile = Token + '/' + Token + 'localeq.csv' # 'atlasexp/Control258localeq.csv'    # <- regular csv from the .nii to csv step

    bright_points = genfromtxt(csvfile, delimiter=',')

    locations = bright_points[:, 0:3]

    regions = [atlas_data[l[0], l[1], l[2]] for l in locations]

    outfile = open(Token + '/' + Token + '.region.csv', 'w')
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

    newToken = Token + '.region'
    atlas = atlasregiongraph(newToken, Token)
    
    atlas.generate_atlas_region_graph(None, numRegions)
