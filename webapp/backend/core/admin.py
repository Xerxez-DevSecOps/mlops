from django.contrib import admin

# Register your models here.

from core.models import Sample
admin.site.register(Sample)

from core.models import Insurance

admin.site.register(Insurance)