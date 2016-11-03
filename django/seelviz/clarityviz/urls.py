#!/usr/bin/env python
#-*- coding:utf-8 -*-

from django.conf.urls import url
from . import views	# this imports views.py from the local directory/package

urlpatterns = [
	url(r'^$', views.index, name='index'),
	# url(r'^outputs/', views.outputs, name='outputs'),

	# [a-z] means a through z, the + means any number of digits >= 1ß
	# url(r'^(?P<token>[a-z|A-Z]+)/$', views.output, name='output'),

	url(r'^tokencompute/$', views.token_compute, name='tokencompute'),
	url(r'^download/(?P<path>.*)$', views.download, name='download'),
	url(r'^plot/(?P<path>.*)$', views.plot, name='plot'),
]
