FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

WORKDIR /app/app

COPY ./platform/app /app/app

COPY ./platform/config_map.yaml /app/config_map.yaml

COPY ./platform/requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/app"

SHELL ["/bin/bash", "-c"]

WORKDIR /app/app

ENTRYPOINT python -m uvicorn app.main:app --app-dir /app --host 0.0.0.0 --port 80

