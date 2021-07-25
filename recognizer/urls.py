from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='recognizer-home'),
    path('about', views.about, name='recognizer-about'),
]
