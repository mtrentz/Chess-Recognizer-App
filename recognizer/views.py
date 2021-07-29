from django.shortcuts import render, redirect
from django.contrib import messages
from .models import Board
from .forms import BoardUploadForm
from .chess_recognizer import ChessRecognizer, translate_pred_to_pt, translate_pred_to_unicode
from PIL import Image
import pprint

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

            # Passa um usuário só se estiver logado, caso contrario fica NULL
            if not request.user.is_anonymous:
                board.user = request.user

            # Pega a imagem na memória recém uploadada e passa pra uma img PIL
            stream = board.board_img.file
            img = Image.open(stream)

            # Faz a classificação na img PIL
            recognizer = ChessRecognizer(img)

            # Preenche o resultado da classificação no campo do form, que vai pro DB.
            predicted_board = recognizer.predicted_board
            board.board_matrix = str(predicted_board)
            board.save()
            messages.success(request, f'Uploaded board image!')
            
            unicode_matrix = translate_pred_to_unicode(predicted_board).tolist()
            context = {
                'form': form,
                'unicode_matrix': unicode_matrix,
            }
            return render(request, 'recognizer/upload.html', context)

    else:
        form = BoardUploadForm()
        context = {
            'form': form,
            'unicode_matrix': None,
        }
        return render(request, 'recognizer/upload.html', context)