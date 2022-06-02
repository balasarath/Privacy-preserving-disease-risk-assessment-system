from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('hospital',views.hospital,name = 'Hospital'),
    path('',views.patient,name = 'Patient'),
    path('result',views.result,name = 'result'),
    path('putsym',views.putSymptoms,name = 'putSymptoms'),
    path('uploadFile',views.uploadFile,name='uploadFile'),
    path('getFile',views.getFile,name='getFile'),
    path('hospitalresult',views.hospitalResult,name = 'hospitalResult'),
    path('sendFile',views.sendFile,name = 'sendFile'),
    path('reset',views.reset,name = 'reset'),
]
# if settings.DEBUG:
#     urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
# else:
#     urlpatterns += staticfiles_urlpatterns()
