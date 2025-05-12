FROM python:3.9-alpine3.13

LABEL maintainer="xerxez.in"
ENV PYTHONUNBUFFERED 1
COPY ./requirements.txt /requirements.txt
COPY ./backend /backend
WORKDIR /backend
EXPOSE 8000

RUN python -m venv /py && \
    /py/bin/pip install --upgrade pip && \
    /py/bin/pip install -r /requirements.txt && \
    adduser --disabled-password --no-create-home backend

ENV PATH="/py/bin:$PATH"
USER backend

