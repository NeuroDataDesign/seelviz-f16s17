from django.db import models

# Create your models here.
class TokenUpload(models.Model):
    token = models.CharField(max_length = 20)

    num_points = models.CharField(max_length = 20)

    # base_csv_path = models.TextField()
    # nodes_csv_path = models.TextField()
    # edges_csv_path = models.TextField()
    # graphml_path = models.TextField()

    # base_graph_path = models.TextField()
    # density_graph_path = models.TextField()
    # atlas_region_graph_path = models.TextField()

    def __str__(self):
        return self.token

class CsvUpload(models.Model):
    file_name = models.TextField()

    def __str__(self):
        return self.token

class TemplateClass(models.Model):
    key_test = models.ForeignKey(TokenUpload, on_delete = models.CASCADE)

    token = models.CharField(max_length = 20)

    base_csv_path = models.TextField()