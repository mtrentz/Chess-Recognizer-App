# Chess-Recognizer-App

Web app that recognizes and generates FEN code from a board screenshot.

Takes you to Lichess to continue playing.

![Preview](https://media.giphy.com/media/vihO779lBdl7vmCWGz/giphy.gif?cid=790b7611efa28a1e1c9f48a5572ac23572d9354270da3652&rid=giphy.gif&ct=g)

---

# Running with Docker

Building the image
```
docker build -t chess-recognizer .
``` 
Running on port 8000
```
docker run -p 8000:8000 chess-recognizer
```

# Running without Docker
Install all required packages
```
pip install -r requirements.txt
```
which are:
```
Django==3.2.6
django-crispy-forms==1.12.0
numpy==1.21.2
opencv-python==4.5.3.56
Pillow==8.3.1
python-decouple==3.4
torch==1.9.0
torchvision==0.10.0
```

Start the server with:
```
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```
