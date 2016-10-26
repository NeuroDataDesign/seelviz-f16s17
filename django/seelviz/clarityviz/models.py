from django.db import models

# Create your models here.
class TokenUpload(models.Model):
	token = models.TextField()

	def __str__(self):
		return self.token