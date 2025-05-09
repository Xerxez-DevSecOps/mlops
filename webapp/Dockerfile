FROM python:3.13.0-alpine3.20

ENV PYTHONUNBUFFERED 1

WORKDIR /app
COPY ./requirements.txt /requirements.txt

ENV PATH="/py/bin:$PATH"

RUN python -m venv /py && \
    pip install --upgrade pip

RUN pip install -r /requirements.txt

COPY ./backend /backend
WORKDIR /backend 

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

