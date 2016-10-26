from django.shortcuts import render

from django.http import HttpResponse
# Create your views here.
from clarityviz import test

def index(request):
    # return HttpResponse("<h2>Hello World</h2>")
    if(request.GET.get('tokenButton')):
        test.testFunction( request.GET.get('myToken') )
    return render(request, 'clarityviz/index.html')

def outputs(request):
    return render(request, 'clarityviz/outputs.html')
