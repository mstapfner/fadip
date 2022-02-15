import logging

import boto3

from app.algorithms.copod import COPODAlgorithm
from app.algorithms.knn import KNNAlgorithm
from app.algorithms.iforest import IForestAlgorithm
from app.algorithms.cblof import CBLOFAlgorithm
from app.algorithms.hbos import HBOSAlgorithm

from datetime import datetime, timedelta
from app.dependencies.config_dependencies import connect_to_prometheus_instances, load_config
from app.dependencies.aws_dependencies import load_aws_s3_client




from fastapi import BackgroundTasks, APIRouter, Depends
from dask import dataframe as dd
from app.prometheus_requests.prometheus_requests import prom_custom_query

router = APIRouter()


@router.get("/import_train_all")
async def import_train_all(background_tasks: BackgroundTasks,
                           prom_cons: dict = Depends(connect_to_prometheus_instances),
                           s3_client: boto3.client = Depends(load_aws_s3_client),
                           conf: dict = Depends(load_config)):

    """Handles the HTTP Route for /import_train_all and delegates the function to the background queue

    :param background_tasks: The built-in background task queue from FastAPI
    :param prom_cons: Dependence to the Prometheus connection objects
    :param s3_client: Dependence to the AWS S3 client object
    :param conf: Dependence to the config map

    :return: Returns text that the job was started as HTTP Response

    """

    background_tasks.add_task(import_and_train, prom_cons, s3_client, conf)
    return "Started job: import & train all"


def import_and_train(prom_cons: dict, s3_client: boto3.client, conf: dict):

    """Imports data from prometheus and trains all algorithms with the specified data, then stores the corresponding
    models either in S3 or on the local disk (specified in config file)

    :param prom_cons: Prometheus connection object map
    :param s3_client: AWS S3 Client object
    :param conf: Config map

    :return: Returns the results of the training as a map

    """

    # Load timeseries from configuration
    datasources = conf["mapping"]
    training_result = {}

    for datasource in datasources:

        timeseries = datasource["timeseries"]
        datasource_id = datasource["datasource_id"]

        for timeserie in timeseries:

            start_time = timeserie["training_starttime"]
            end_time = timeserie["training_endtime"]
            query_name = timeserie["query"]
            ts_type = timeserie["ts_type"]
            step = 1

            algorithms = timeserie["algorithms"]
            prom_con = prom_cons.get(datasource_id)

            logging.info("Start data import for datasource: {0} and query: {1}".
                         format(datasource_id, query_name))

            """Split complete prometheus timeinterval into 24hour-splits"""
            start_time_dt = datetime.fromtimestamp(start_time)
            end_time_dt = datetime.fromtimestamp(end_time)
            days = (end_time_dt - start_time_dt).days

            """Importing the prometheus data split by split"""

            # First time slot starts with start_time
            time_slot_start = start_time_dt
            dataframes = []
            for day in range(days + 1):
                time_slot_end = time_slot_start + timedelta(days=1)

                # Check if first_slot_end is bigger then the final time end, then replace it
                if time_slot_end > end_time_dt:
                    time_slot_end = end_time_dt

                # Request timeslot from Prometheus
                metrics = []
                try:
                    metrics = prom_custom_query(prom_con, query_name, time_slot_start, time_slot_end, ts_type, step)
                except ConnectionError as conn_error:
                    # Timeout or other request problem when trying to connect to prometheus
                    logging.info(conn_error)
                    # Jump to next timeseries
                    break

                if len(metrics) > 0:
                    dataframes.append(metrics)
                time_slot_start = time_slot_end

            # Merge dataframes into dask dataframes
            dask_df = None
            if len(dataframes) > 0:
                dask_df = dd.from_pandas(dataframes[0], chunksize=100)
                for x in range(len(dataframes) - 1):
                    temp_df = dd.from_pandas(dataframes[x + 1], chunksize=100)
                    dask_df = dd.merge(dask_df, temp_df, how='outer', on=['timestamp', 'value'])
            else:
                logging.info("Skip training of {0}, all dataframes of query {1} in timeinterval {2} - {3} are empty".
                             format(datasource_id, query_name, start_time_dt, end_time_dt))
                break


            amount_features = len(dask_df.columns.tolist())

            # Start the training of the algorithms
            for algorithm in algorithms:
                clf = None
                success = False
                algorithm_name = algorithm["id"]
                identity = datasource_id + "*+*" + timeserie["id"] + "*+*" + algorithm_name
                contamination_train = algorithm["contamination_train"]

                # Extension of Algorithms: Add the algorithm identifier (can be freely chosen, but must be unique) here and
                #   the corresponding algorithm class and training methods (that was added to the `algorithms` folder)
                try:
                    if algorithm_name == "iforest":
                        clf = IForestAlgorithm(contamination=contamination_train, ts_type=ts_type, features=amount_features, n_jobs=2)
                        clf.train_algorithm_unsupervised(dask_df)
                    elif algorithm_name == "cblof":
                        clf = CBLOFAlgorithm(contamination=contamination_train)
                        clf.train_algorithm_unsupervised(dask_df)
                    elif algorithm_name == "hbos":
                        clf = HBOSAlgorithm(contamination=contamination_train)
                        clf.train_algorithm_unsupervised(dask_df)
                    elif algorithm_name == "knn":
                        clf = KNNAlgorithm(contamination=contamination_train)
                        clf.train_algorithm_unsupervised(dask_df)
                    elif algorithm_name == "copod":
                        clf = COPODAlgorithm(contamination=contamination_train)
                        clf.train_algorithm_unsupervised(dask_df)
                    elif algorithm_name == "oneclasssvm":
                        clf = OneClassSVMAlgorithm(contamination=contamination_train)
                        clf.train_algorithm_unsupervised(dask_df)
                    elif algorithm_name == "prophet":
                        # Adjustments to FBProphet
                        dask_df["timestamp"] = dask_df.index
                        # Convert to pandas
                        dask_df = dask_df.compute()
                        dask_df.rename(columns={"timestamp": "ds", "value": "y"}, inplace=True)
                        clf = ProphetAlgorithm()
                        clf.train_algorithm_unsupervised(dask_df)

                    # Store the model as a file either to S3 or local disk
                    if clf is not None:
                        if conf["model_storage"]["s3"]["activated"]:
                            success = clf.store_model_to_s3(name=identity,
                                                            bucket=conf["model_storage"]["s3"]["bucket_name"],
                                                            client=s3_client)
                        else:
                            success = clf.store_model_to_file(identity)

                    if success:
                        training_result[identity] = True
                        logging.info("Successfully trained {0} for query: {1} and stored data model.".
                                     format(algorithm_name, query_name))
                    else:
                        training_result[identity] = False
                except FileNotFoundError as file_not_found_error:
                    logging.info(file_not_found_error)
                    logging.info("Failed to train algorithm: %s", algorithm_name)
                    training_result[identity] = False

    logging.info("Finished job: import & train all")
    return training_result
