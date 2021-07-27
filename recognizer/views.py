from django.shortcuts import render, redirect
from django.contrib import messages
from .models import Board
from .forms import BoardUploadForm


# Create your views here.
def home(request):
    context = {
        'data': Board.objects.all()
    }

    return render(request, 'recognizer/home.html', context)

# Create your views here.
def about(request):
    context = {
        'data': Board.objects.all()
    }
    return render(request, 'recognizer/about.html', context)

def upload(request):
    if request.method == 'POST':
        form = BoardUploadForm(request.POST, request.FILES)

        if form.is_valid():
            board = form.save(commit=False)
            if not request.user.is_anonymous:
                board.user = request.user
            board.save()
            messages.success(request, f'Uploaded board image!')
            return redirect('recognizer-upload')

    else:
        form = BoardUploadForm()

    context = {
        'form': form
    }

    return render(request, 'recognizer/upload.html', context)