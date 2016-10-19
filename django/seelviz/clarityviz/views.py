from django.shortcuts import render

from django.http import HttpResponse
# Create your views here.

def index(request):
	# return HttpResponse("<h2>Hello World</h2>")
	return render(request, 'clarityviz/index.html')
