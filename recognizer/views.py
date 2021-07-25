from django.shortcuts import render
from django.http import HttpResponse
from .models import Board


# Create your views here.
def home(requests):
    context = {
        'data': Board.objects.all()
    }

    return render(requests, 'recognizer/home.html', context)

# Create your views here.
def about(requests):
    context = {
        'data': Board.objects.all()
    }
    return render(requests, 'recognizer/about.html', context)