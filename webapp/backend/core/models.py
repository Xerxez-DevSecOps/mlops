from django.db import models

# Create your models here.

class Sample(models.Model):
    """ Sample model """
    attachment = models.FileField()

class Insurance(models.Model):
    MALE = 1
    FEMALE = 0
    SEX_CHOICES = (
        (MALE, 'Male'),
        (FEMALE, 'Female')
    )

    SOUTHEAST = 0
    SOUTHWEST = 1
    NORTHEAST = 2
    NORTHWEST = 3
    REGION_CHOICES = (
        (NORTHWEST, 'Northwest'),
        (NORTHEAST, 'Northeast'),
        (SOUTHWEST, 'Southwest'),
        (SOUTHEAST, 'Southeast'),
    )

    age = models.FloatField()
    sex = models.IntegerField(choices=SEX_CHOICES)
    bmi = models.FloatField()
    children = models.FloatField()
    smoker = models.IntegerField(choices=((0, 'Smoker'),(1, 'non-Smoker')))
    region = models.IntegerField(choices=REGION_CHOICES)

