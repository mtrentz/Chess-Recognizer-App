from django.shortcuts import render, redirect
from django.contrib import messages
from .models import Board
from .forms import BoardUploadForm
from django.views.generic import CreateView


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


class BoardCreateView(CreateView):
    model = Board
    fields = ['board_img', 'board_matrix', 'fen']

    def form_valid(self, form):
        form.instance.board_img = self.request.FILES
        form.instance.user = self.request.user
        return super().form_valid(form)


def upload(request):
    # board_form = BoardUploadForm(request.POST, request.FILES)
    # if board_form.is_valid():
    #     board_form.save()
    #     messages.success(request, f'Uploaded board image!')
    #     return redirect('recognizer-upload')
    # else:
    #     messages.error(request, f'Nao valido!')

    if request.method == 'POST':
        form = BoardUploadForm(request.POST, request.FILES)
        #board_form.fields['user'] = request.user

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