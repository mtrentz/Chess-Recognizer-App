from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload, name='recognizer-home'),
    # path('about/', views.about, name='recognizer-about'),
    # path('upload/', views.upload, name='recognizer-upload'),
]
