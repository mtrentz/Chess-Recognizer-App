from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Board(models.Model):
    fen = models.TextField()
    date_uploaded = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.RESTRICT)

    def __str__(self):
        return self.fen
