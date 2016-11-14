from django.db import models
from django.core.urlresolvers import reverse

import glob
import os
import os.path

# Create your models here.
class Compute(models.Model):
    token = models.CharField(max_length = 30)
    orientation = models.CharField(max_length = 3)
    num_points = models.CharField(max_length = 20)


    # base_csv_path = models.TextField()
    # nodes_csv_path = models.TextField()
    # edges_csv_path = models.TextField()
    # graphml_path = models.TextField()

    # base_graph_path = models.TextField()
    # density_graph_path = models.TextField()
    # atlas_region_graph_path = models.TextField()


    def get_absolute_url(self):
        # query_set = Compute.objects.filter(pk=self.pk)
        # for compute in query_set:
        #     new_compute = compute
        #
        # print(new_compute.token)
        token = self.token + 'reorient_atlas'

        plotly_files = []
        all_files = []
        for filepath in glob.glob('output/' + token + '/*'):
            absPath = os.path.abspath(filepath)
            if not os.path.isdir(absPath):
                filename = filepath.split('/')[2]
                all_files.append(filename)
                if filepath.endswith('html'):
                    plotly_files.append(filename)
        context = {'token': token, 'all_files': all_files, 'plotly_files': plotly_files}
        return reverse('clarityviz:output', kwargs=context)
        # return reverse('clarityviz:output', kwargs={'pk': self.pk})

    def __str__(self):
        return self.token

class Plot(models.Model):
    token_compute = models.ForeignKey(Compute, on_delete=models.CASCADE)
    plot_type = models.TextField()

class CsvUpload(models.Model):
    file_name = models.TextField()

    def __str__(self):
        return self.token

class TemplateClass(models.Model):
    key_test = models.ForeignKey(Compute, on_delete=models.CASCADE)

    token = models.CharField(max_length = 20)

    base_csv_path = models.TextField()