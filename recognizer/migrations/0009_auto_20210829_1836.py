# Generated by Django 3.2.6 on 2021-08-29 18:36

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('recognizer', '0008_alter_board_user'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='board',
            name='date_uploaded',
        ),
        migrations.RemoveField(
            model_name='board',
            name='user',
        ),
    ]
