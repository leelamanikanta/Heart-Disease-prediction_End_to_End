FROM python:3.8-slim-buster
COPY requirements.txt /cardioapp/requirements.txt
WORKDIR /cardioapp
RUN pip install -r /cardioapp/requirements.txt
COPY . /cardioapp
CMD python app.py