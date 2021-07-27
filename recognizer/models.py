from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

# Create your models here.
class Board(models.Model):
    board_img = models.ImageField(upload_to='board_img')
    board_matrix = models.TextField(default=None, blank=True, null=True)
    fen = models.TextField(default=None, null=True, blank=True)
    date_uploaded = models.DateTimeField(default=timezone.now)
    user = models.ForeignKey(User, on_delete=models.RESTRICT, blank=True, null=True)
    #user = models.ForeignKey(User, on_delete=models.CASCADE)

    # def __str__(self):
    #     return self.board_matrix_unicode
