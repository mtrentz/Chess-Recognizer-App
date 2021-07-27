from django.urls import path
from .views import BoardCreateView
from . import views

urlpatterns = [
    path('', views.home, name='recognizer-home'),
    path('about/', views.about, name='recognizer-about'),
    #path('upload/', BoardCreateView.as_view(), name='board-create'),
    path('upload/', views.upload, name='recognizer-upload'),
]
