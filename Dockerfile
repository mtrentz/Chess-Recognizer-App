FROM pytorch/pytorch
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt
RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY . /app/
RUN python manage.py migrate
RUN python manage.py makemigrations
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]