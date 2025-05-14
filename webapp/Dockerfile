FROM python:3.9-alpine3.13

LABEL maintainer="xerxez.in"
ENV PYTHONUNBUFFERED 1

COPY ./backend /backend
WORKDIR /backend
COPY ./requirements.txt /requirements.txt


RUN python -m venv /py && \
    /py/bin/pip install --upgrade pip && \
    apk add --update --no-cache postgresql-client && \
    apk add --update --no-cache --virtual .tmp-deps \
        build-base \
        postgresql-dev \
        musl-dev \
        linux-headers && \
    /py/bin/pip install -r /requirements.txt && \
    apk del .tmp-deps && \
    adduser --disabled-password --no-create-home backend && \
    chown -R backend /backend

ENV PATH="/py/bin:$PATH"
USER backend
EXPOSE 8000

