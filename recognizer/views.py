from django.shortcuts import render, redirect
from django.contrib import messages
from .models import Board
from .forms import BoardUploadForm
from .chess_recognizer import ChessRecognizer, translate_pred_to_unicode, board_to_fen
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

    context = {
        'form': None,
        'unicode_matrix': None,
        'white': None,
        'black': None,
        'fen': None,
        'lichess_urls': None,
    }

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
            # Pega o FEN do board, default pra white como a nova jogada
            fen_white = board_to_fen(predicted_board)
            # Copia o mesmo pro preto e muda a parte q diz quem é o prox
            fen_black = fen_white[:-12] + 'b' + fen_white[-11:]

            lichess_play_white_url = "https://lichess.org/analysis/standard/" + fen_white
            lichess_play_black_url = "https://lichess.org/analysis/standard/" + fen_black

            context['form'] = form
            context['unicode_matrix'] = unicode_matrix
            context['fen'] = {
                'fen_white': fen_white,
                'fen_black': fen_black,
            }
            context['lichess_urls'] = {
                'play_white_url': lichess_play_white_url,
                'play_black_url': lichess_play_black_url,
            }

            return render(request, 'recognizer/upload.html', context)

    else:
        form = BoardUploadForm()

        context['form'] = form

        return render(request, 'recognizer/upload.html', context)