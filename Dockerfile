FROM pytorch/pytorch
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY docker-requirements.txt /app/
RUN pip install -r docker-requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY . /app/
RUN python manage.py makemigrations
RUN python manage.py migrate
EXPOSE 8000
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]