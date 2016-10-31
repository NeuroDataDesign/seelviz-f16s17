from django.shortcuts import render
# from django.template import loader
from django.http import HttpResponse
from .models import TokenUpload
# Create your views here.
from . import test

def index(request):
    # return HttpResponse("<h2>Hello World</h2>")

    # get token from user form, then pass that token into the url 
    # while running the pipeline, then open the link to the output 
    # with the token e.g. /clarityviz/fear199

    # template = loader.get_template('clarityviz/index.html')

    # return HttpResponse(template.render(context, request))

    uploads = TokenUpload.objects.all()
    context = {'uploads': uploads}

    if(request.GET.get('tokenButton')):
        test.testFunction( request.GET.get('myToken') )
    return render(request, 'clarityviz/index.html', context)

def output(request, token):
    return render(request, 'clarityviz/outputs.html')
