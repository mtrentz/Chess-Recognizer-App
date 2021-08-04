from django import forms
from .models import Board

class BoardUploadForm(forms.ModelForm):
    class Meta:
        model = Board
        fields = ['board_img']
    
    def __init__(self, *args, **kwargs):
        super(BoardUploadForm, self).__init__(*args, **kwargs)
        self.fields['board_img'].label = 'Board Image'
