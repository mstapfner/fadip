FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

WORKDIR /app/app

COPY ./app /app/app

COPY config_map.yaml /app/config_map.yaml

COPY requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/app"

SHELL ["/bin/bash", "-c"]

WORKDIR /app/app

ENTRYPOINT python -m uvicorn app.main:app --app-dir /app --host 0.0.0.0 --port 80

