#!/usr/bin/env python
#-*- coding:utf-8 -*-

from django.conf.urls import url
from . import views	# this imports views.py from the local directory/package

urlpatterns = [
    url(r'^$', views.index, name='index'),
    # url(r'^outputs/', views.outputs, name='outputs'),

    # [a-z] means a through z, the + means any number of digits >= 1ß
    # url(r'^(?P<token>[a-z|A-Z]+)/$', views.output, name='output'),

    url(r'^log/$', views.LogView.as_view(), name='log'),
    url(r'^(?P<pk>[0-9]+)$', views.OutputView.as_view(), name='output'),
    url(r'^compute/$', views.ComputeCreate.as_view(), name='compute'),

    url(r'^tokencompute/$', views.token_compute, name='tokencompute'),
    url(r'^download/(?P<file_name>.*)$', views.download, name='download'),
    url(r'^plot/(?P<file_info>.*)$', views.plot, name='plot'),
]
