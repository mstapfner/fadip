import sys
import os

from prometheus_api_client import PrometheusConnect, PrometheusApiClientException
from prometheus_pandas import query
from requests import RequestException, ConnectionError
from jsonschema import validate, ValidationError
import logging
import yaml
from urllib3.exceptions import NewConnectionError


def load_config():

    """Loads the config map out of the file config_map.yaml and call validation

   :return: Returns the read config map as dict

    """

    config_map = []
    try:
        with open("../config_map.yaml") as f:
            config_map = yaml.safe_load(f)
#             validate_config(config_map)
    except FileNotFoundError as file_not_found_error:
        logging.error(file_not_found_error)
        raise FileNotFoundError("config_map.yaml was not found, is mandatory for running application")

    config_map = config_map["fadip"]
    return config_map


def validate_config(config_map):

    """Validates the given config map with the specified scheme, stops application when invalid

    :param config_map: Config Map read from the file "config_map.yaml"
    :type config_map: dict

    """

    try:
        with open("../config_schema.json") as schema:
            validate(config_map, yaml.load(schema, yaml.SafeLoader))
    except FileNotFoundError as file_not_found_error:
        logging.info(file_not_found_error)
        raise FileNotFoundError("The json scheme for validation could not be found")
    except ValidationError as validation_error:
        logging.info(validation_error)
        raise ValidationError("The given config file is not valid")

    config_map = config_map["fadip"]

    if config_map["version"] != 0.1:
        raise ValueError("Wrong version of config file")
    if config_map["working_mode"] != "normal":
        raise ValueError("Currently, only working_mode normal is implemented")
    if len(config_map["datasources"]) == 0:
        raise ValueError("Datasources can't be empty")


def check_prometheus_instances(config_map):

    """Checks the connection state of the different Prometheus instances configured in Config map

    :param config_map: Config Map read from the file "config_map.yaml"
    :type config_map: dict

    :return: returns the connection objects of the prometheus instances
    """

    prometheus_instances = []
    for datasource in config_map["datasources"]:
        if datasource["type"] == "prometheus":
            try:
                prom = PrometheusConnect(url=datasource["url"], disable_ssl=datasource["disable_ssl"])
                prom.check_prometheus_connection()
                prometheus_instances.append(prom)
            except (PrometheusApiClientException, RequestException, ConnectionError, NewConnectionError) as \
                    prometheus_api_client_exception:
                logging.warning("failed to connect to prometheus instance {0}  with url {1}".
                                format(datasource["id"], datasource["url"]))

    return prometheus_instances


# Prometheus Connection
def connect_to_prometheus_instances():

    """Connects to the configured prometheus instances from the config maps

    :return: returns the prometheus connection objects as a map with prometheus instance id as key and connection as
    value

    """

    conf = load_config()
    prometheus_instances = {}
    for datasource in conf["datasources"]:
        if datasource["type"] == "prometheus":
            try:
                prom = query.Prometheus(datasource["url"])
                prometheus_instances[datasource["id"]] = prom
            except PrometheusApiClientException as prometheus_api_client_exception:
                logging.info("failed to connect to prometheus, there is no instance reachable under the given "
                             "prometheus uri")
                logging.error(prometheus_api_client_exception)
            except RequestException as request_exception:
                logging.info("failed to connect to prometheus. ")
                logging.error(request_exception)

    return prometheus_instances
