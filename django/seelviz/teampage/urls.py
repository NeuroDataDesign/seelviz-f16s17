from django.conf.urls import url
from . import views	# this imports views.py from the local directory/package

urlpatterns = [
	url(r'^$', views.index, name='index'),
]
