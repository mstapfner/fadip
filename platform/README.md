# Dev setup using Conda

Create new conda environment: `conda env create -f environment.yml --name platform`

Activate environment: `conda activate flex-adf`

Export the Pythonpath `export PYTHONPATH=..<PATH>../fadip/platform`

Change Working directory in PyCharm run configurations to /platform/

Place the Datasets inside the `./platform/datasets` folder, more information on that in the `README.md` in 
`./platform/datasets`

Start Development Env: `docker-compose -f docker-compose-dev.yml up -d`

Start the script via: `python app/main.py`

# Prod setup 

Build flex-adf service: `docker buildx build --output type=docker --platform linux/amd64 -t fadip:latest .`

Start the environment `docker-compose up`

## Setup with PyCharm:

https://stackoverflow.com/questions/28326362/pycharm-and-pythonpath

## Config file specification: 
```yaml
# FADIP Configuration files

fadip:
  version: 0.1
  inital_setup: True | False # When true, perform the inital setup for model training
  working_mode: "normal" # for future working modes
  datasources: # all datasources that should be included 
    - type: "prometheus" 
      id: "prom1" # Unique, To support multiple datasources
      url: "http://localhost:9090"
      username: "" # Optional
      password: "" # Optional
      disable_ssl: True | False # Optional
    - type: "open-data"
      id: "opendata1" # Unique, to support multiple datasources
      location: "<filepath>" # Dataset location Filepath
      format: "" # Dataset format
      name: "" # Dataset name
    - ...
  alerting: 
    slack: 
      # TBD
      url: "" # Url of the slack instance
      webhook_token: "" # Token of the slack instance
    teams: 
      # TBD
      url: "" # Url of the teams instance
      webhook_token: "" Token of the teams instance
  algorithms:
    - id: "" # identifier for the algorithms
      built_in: true # or false, whether the algorithm is implemented in the flex-adf_old adapater
      secret: "" # secret for secure communicaton with the flex-adf_old compatible adapter
      url: ""
    - ...
  mapping: # Mapping of datasources and timeseries 
    - datasource_id: "" # Id of the datasources from above
      timeseries: 
        - id: "" # identifier for timeseries
          query: "process_cpu_seconds_total" # PromQL query for inclusion in anomaly detection
          chunk_size: "5m" # Timeseries chunk size
          algorithms: ["id_1", "id_2"] # ids of the algorithms
          alerting: true # | false 
    - datasource_id: "" # Id of the datasources from above
      timeseries: 
        - id: "" # identifier for timeseries
          query: "process_cpu_seconds_total" # PromQL query for inclusion in anomaly detection
          chunk_size: "5m" # Timeseries chunk size
          algorithms: ["id_1", "id_2"] # ids of the algorithms
          alerting: true # | false 
    - ...

```

## Note for using Dask Juypter Notebook: 

(Apple Silicon safe)

```shell
git clone https://github.com/dask/dask-docker

export DOCKER_BUILDKIT=0
export COMPOSE_DOCKER_CLI_BUILD=0

cd dask-docker 
docker-compose build
docker-compose up 
```

goto 0.0.0.0:8888 to find notebook online 


## Note for Apple Silicon Chips (M1)

Installation of SciPy:
https://stackoverflow.com/questions/65745683/how-to-install-scipy-on-apple-silicon-arm-m1 

Installation of Sci-Kit Learn: 
- Use Conda (from miniforge: https://github.com/conda-forge/miniforge)
- Install and Create new conda environment
- Activate environment
- Install all packages that are available for conda: 
  
  `$ while read requirement; do conda install --yes $requirement; done < requirements.txt`
- Install all missing/ignored packages via pip

Installation of TensorFlow: https://developer.apple.com/metal/tensorflow-plugin/ 

### General Apple Silicon Limitations
https://stackoverflow.com/questions/65095614/macbook-m1-and-python-libraries


