FROM continuumio/miniconda3 AS build

COPY environment.yml .
RUN conda env create -f environment.yml

RUN conda install -c conda-forge conda-pack

RUN conda-pack -n conda-flex-adf -o /tmp/env.tar && \
  mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
  rm /tmp/env.tar

RUN /venv/bin/conda-unpack


FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8 AS runtime

COPY --from=build /venv /venv

WORKDIR /app/app

COPY ./app /app/app

COPY config_map.yaml /app/config_map.yaml

ENV PYTHONPATH "${PYTHONPATH}:/app"

SHELL ["/bin/bash", "-c"]

WORKDIR /app/app

ENTRYPOINT source /venv/bin/activate && python -m uvicorn app.main:app --app-dir /app --host 0.0.0.0 --port 80
            # "while True: print('lol')"

