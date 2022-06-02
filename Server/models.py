from django.db import models
from .storage import OverwriteStorage

class Symptoms(models.Model):
    Sym_id = models.AutoField(primary_key=True)
    Sym_name = models.CharField(max_length = 150,null=True)
    def __str__(self):
        s = " id : {}, Symptom name: {}".format(self.Sym_id,self.Sym_name)
        return s

class FileModel(models.Model):
    upload = models.FileField(storage = OverwriteStorage(), upload_to = 'uploads/')

class ResultImg(models.Model):
    img = models.FileField(storage = OverwriteStorage(), upload_to = 'Result/')
    
