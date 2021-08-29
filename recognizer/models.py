from django.db import models
from django.utils import timezone

# Create your models here.
class Board(models.Model):
    board_img = models.ImageField(upload_to='board_img')
    board_matrix = models.TextField(default=None, blank=True, null=True)
    fen = models.TextField(default=None, null=True, blank=True)
