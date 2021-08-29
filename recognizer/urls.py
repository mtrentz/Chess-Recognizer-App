from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload, name='recognizer-home'),
]
