from django import forms
from .models import Board

class BoardUploadForm(forms.ModelForm):
    class Meta:
        model = Board
        fields = ['board_img']