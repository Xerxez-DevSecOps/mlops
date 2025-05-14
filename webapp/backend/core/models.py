from django.db import models

# Create your models here.

class Sample(models.Model):
    """ Sample model """
    attachment = models.FileField()
