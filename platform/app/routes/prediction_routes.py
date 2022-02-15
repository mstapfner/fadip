import copy
import logging
import collections
import os
from typing import Optional

import boto3

from datetime import datetime, timedelta

import numpy as np
import prometheus_pandas.query
from algorithms.copod import COPODAlgorithm
from app.algorithms.hbos import HBOSAlgorithm
from app.algorithms.knn import KNNAlgorithm
from app.dependencies.config_dependencies import connect_to_prometheus_instances, load_config
from app.dependencies.aws_dependencies import load_aws_s3_client
from app.algorithms.iforest import IForestAlgorithm
from app.algorithms.cblof import CBLOFAlgorithm
from app.models.anomaly import Anomaly
from app.integrations.slack_integration import send_anomaly_to_slack
from app.dependencies.database_dependencies import load_session
from app.integrations.msteams_integration import send_anomaly_to_msteams
from sqlalchemy.orm import session

from fastapi import BackgroundTasks, APIRouter, Depends
from app.prometheus_requests.prometheus_requests import prom_custom_query
from app.utils.graph_builder import build_anomaly_graph_alerting

router = APIRouter()


@router.get("/predict_timeslot")
async def predict_timeslot(background_tasks: BackgroundTasks,
                           debugging_args: Optional[str] = None,
                           prom_con: dict = Depends(
                               connect_to_prometheus_instances),
                           db_session: session = Depends(load_session),
                           s3_client: boto3.client = Depends(load_aws_s3_client),
                           conf: dict = Depends(load_config)):
    """Handles the HTTP Route for /predict_timeslot and delegates the function to the background queue

    :param debugging_args: Debugging arguments for manually adding anomalies to the result
    :param background_tasks: The built-in background task queue from FastAPI
    :param prom_con: Dependence to the Prometheus connection objects
    :param db_session: Dependence to the database session object
    :param s3_client: Dependence to the AWS S3 client object
    :param conf: Dependence to the config map

    :return: Returns text that the job was started as HTTP Response

    """

    background_tasks.add_task(predict_all_with_timeslot, debugging_args, prom_con, db_session, s3_client, conf)
    return "Started job: predict all with timeslot"


def predict_all_with_timeslot(debugging_args, prom_cons: dict,
                              db_session: session, s3_client: boto3.client, conf: dict):
    """Predicts the last 3 hours of data from prometheus with the model (from the specified algorithms) for anomalies
    and stores them in the database.

    :param debugging_args: debugging arguments for manually adding anomalies to the result
    :param prom_cons: Prometheus connection objects
    :param db_session:
    :param s3_client: AWS S3 Client object
    :param conf: Config map

    """

    datasources = conf["mapping"]

    for datasource in datasources:

        timeseries = datasource["timeseries"]
        datasource_id = datasource["datasource_id"]

        for timeserie in timeseries:

            # last 3 hours time-window
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=3)
            query_name = timeserie["query"]
            step = 100
            ts_type = timeserie["ts_type"]
            algorithms = timeserie["algorithms"]

            prom_con = prom_cons.get(datasource_id)

            # Request timeslot from Prometheus
            metrics = prom_custom_query(prom_con, query_name, start_time, end_time, ts_type, step)
            amount_features = len(metrics.columns.tolist())

            for algorithm in algorithms:
                algorithm_name = algorithm["id"]
                contamination_train = algorithm["contamination_train"]
                cache_metrics = copy.deepcopy(metrics)
                # Load trained algorithm from filesystem
                identity = datasource_id + "*+*" + timeserie["id"] + "*+*" + algorithm_name
                clf = None

                # Extension of Algorithms: Add the algorithm identifier (can be freely chosen, but must be unique) here and
                #   the corresponding algorithm class and training methods (that was added to the `algorithms` folder)
                try:
                    if algorithm_name == "iforest":
                        clf = IForestAlgorithm(contamination=contamination_train, features=amount_features,
                                               ts_type=ts_type, n_jobs=2)
                    elif algorithm_name == "cblof":
                        clf = CBLOFAlgorithm(contamination=contamination_train)
                    elif algorithm_name == "hbos":
                        clf = HBOSAlgorithm(contamination=contamination_train)
                    elif algorithm_name == "knn":
                        clf = KNNAlgorithm(contamination=contamination_train)
                    elif algorithm_name == "copod":
                        clf = COPODAlgorithm(contamination=contamination_train)

                    if clf is not None:
                        if conf["model_storage"]["s3"]["activated"]:
                            clf.load_model_from_s3(name=identity, bucket=conf["model_storage"]["s3"]["bucket_name"],
                                                   client=s3_client)
                        else:
                            clf.load_model_from_file(identity)

                        prediction = None
                        prediction_outlier_scores = None
                        if algorithm_name == "prophet":
                            prediction = clf.predict_sample(cache_metrics, ts_type, True)
                        else:
                            prediction, prediction_outlier_scores = clf.predict_sample(cache_metrics, ts_type, True)

                        """Manually insert anomalies if debugging arg is given"""
                        if debugging_args == "add anomalies":
                            if len(prediction) > 5:
                                prediction[1] = 0
                                prediction[2] = 0

                        # Merge anomaly scores and
                        # Important notice: 0 means it is an anomaly, 1 means it is no anomaly
                        # metrics["anomaly score"] = prediction_outlier_scores

                        if algorithm_name == "prophet":
                            cache_metrics["anomaly"] = prediction["anomaly"]
                        else:
                            cache_metrics["anomaly"] = prediction

                        # Check for any anomalies found
                        result = collections.Counter(prediction)

                        if (result.get(0) is not None) and (result.get(0) > 0) and (result.get(1) is not None):
                            logging.info("Detected some anomalies")

                            # Calculate percentage of anomalies of sample
                            percent = (result.get(0) / (result.get(1) + result.get(0))) * 100

                            # Get the first & last anomaly timestamp
                            cache_metrics["time_cache"] = cache_metrics.index
                            cache_metrics["time_cache"] = cache_metrics["time_cache"].apply(lambda x: x.timestamp())

                            first_anomaly_timestamp = cache_metrics[cache_metrics["anomaly"] == 0].min()
                            last_anomaly_timestamp = cache_metrics[cache_metrics["anomaly"] == 0].max()

                            anomaly = Anomaly(
                                query_name=query_name,
                                datasource_id=datasource_id,
                                detected_by=algorithm_name,
                                detection_date=datetime.now(),
                                start_investigated_datetime=start_time,
                                end_investigated_datetime=end_time,
                                first_anomaly_datetime=datetime.fromtimestamp(int(first_anomaly_timestamp['time_cache'])),
                                last_anomaly_datetime=datetime.fromtimestamp(int(last_anomaly_timestamp['time_cache'])),
                                valid=False,
                                checked=False
                            )

                            result = db_session.add(anomaly)
                            commit_result = db_session.commit()

                            file_path = build_anomaly_graph_alerting(query_name, algorithm_name, cache_metrics,
                                                                     identity, ts_type)

                            # Send anomaly to alerting channels if desired
                            if "alerting" in timeserie:
                                # Check for Slack integration
                                if "slack" in conf["alerting"] and "oauth_token" in conf["alerting"]["slack"]:
                                    send_anomaly_to_slack(anomaly, file_path,
                                                          conf["alerting"]["slack"]["oauth_token"])

                                if "teams" in conf["alerting"] and "webhook_url" in conf["alerting"]["teams"]:
                                    result = send_anomaly_to_msteams(anomaly, file_path,
                                                                     conf["alerting"]["teams"]["webhook_url"])
                                    logging.info(result)

                            # remove file from cache
                            os.remove("cache/" + identity + ".png")
                        else:
                            logging.info("No anomalies detected")

                except Exception as e:
                    logging.warning(e)
                    logging.info("no prediction possible, since model file could not be read / error predicting")

    logging.info("Finished job: predict all with timeslot")
    return
