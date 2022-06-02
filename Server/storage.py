from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
import shutil
from tempfile import NamedTemporaryFile
import csv
import time
import psutil

# def has_handle(fpath):
#     for proc in psutil.process_iter():
#         try:
#             for item in proc.open_files():
#                 if fpath == item.path:
#                     return True
#         except Exception:
#             pass

#     return False
class OverwriteStorage(FileSystemStorage):
    def get_available_name(self, name,max_length = None):
        # if self.exists(name):
        #     while has_handle(os.path.join(settings.MEDIA_ROOT, name)) == True:
        #         pass
        os.remove(os.path.join(settings.MEDIA_ROOT, name))
        return name