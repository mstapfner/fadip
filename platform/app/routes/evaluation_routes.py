import glob
import logging
import os
from datetime import datetime

import numpy
import pandas
import pandas as pd
import prometheus_pandas.query

from app.algorithms.copod import COPODAlgorithm
from app.algorithms.loda import LODAAlgorithm
from app.algorithms.lmdd import LMDDAlgorithm
from app.algorithms.autoencoder import AutoEncoderAlgorithm
from app.algorithms.knn import KNNAlgorithm
from app.algorithms.cblof import CBLOFAlgorithm
from app.algorithms.hbos import HBOSAlgorithm
from app.algorithms.iforest import IForestAlgorithm
from app.dependencies.config_dependencies import connect_to_prometheus_instances, load_config
from app.dependencies.database_dependencies import load_session
from fastapi import BackgroundTasks, Depends, APIRouter
from matplotlib import pyplot as plt
# from prometheus_client import Counter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import session
from app.utils.graph_builder import build_comparison_graphs
from app.utils.kmeans_utils import get_distance_by_point
from app.utils.performance_analysis import persist_evaluation, analyze_performance, prepare_and_store_dataframe

router = APIRouter()

DT_FORMAT = '%Y-%m-%dT%H:%M:%SZ'
DT_FORMAT_ALTERNATIVE = '%Y-%m-%d %H:%M:%S'

microsoft_datasets = ["ms_ecommerce-api-incoming-rps", "ms_mongodb-application-rpsv",
                      "ms_middle-tier-api-dependency-latency-uv", "ms_middle-tier-api-dependency-latency-mv",
                      "ms_data-ingress-rate-01", "ms_data-ingress-rate-04", "ms_data-ingress-rate-05"]

microsoft_datasets_other_dt_format = ["ms_data-app-crash-rate2-01", "ms_data-app-crash-rate2-02",
                                      "ms_data-app-crash-rate2-03", "ms_data-app-crash-rate2-04",
                                      "ms_data-app-crash-rate2-05", "ms_data-app-crash-rate2-06",
                                      "ms_data-app-crash-rate2-07", "ms_data-app-crash-rate2-08",
                                      "ms_data-app-crash-rate2-09", "ms_data-app-crash-rate2-10",
                                      "ms_data-app-crash-rate1-01", "ms_data-app-crash-rate1-02",
                                      "ms_data-app-crash-rate1-03", "ms_data-app-crash-rate1-04",
                                      "ms_data-app-crash-rate1-05", "ms_data-app-crash-rate1-06",
                                      "ms_data-app-crash-rate1-07", "ms_data-app-crash-rate1-08",
                                      "ms_data-app-crash-rate1-09"]

commercial_datasets = ["reg1-app18.csv", "reg0-app10.csv", "reg1-app0.csv", "reg1-app1.csv", "reg1-app2.csv",
                       "reg1-app3.csv", "reg1-app4.csv", "reg1-app5.csv", "reg1-app6.csv", "reg1-app7.csv",
                       "reg1-app8.csv", "reg1-app9.csv", "reg1-app10.csv", "reg1-app11.csv", "reg1-app12.csv",
                       "reg1-app13.csv", "reg1-app14.csv", "reg1-app15.csv", "reg1-app16.csv", "reg1-app17.csv",
                       "reg2-app1.csv", "reg2-app3.csv", "reg2-app4.csv", "reg2-app5.csv", "reg2-app8.csv",
                       "reg2-app10.csv", "reg2-app11.csv", "reg2-app17.csv", "reg2-app18.csv"]

nab_datasets = ["art_daily_no_jump", "art_daily_flatmiddle", "ec2_request_latency_system_failure"]

# Extension of Datasets: Add the dataset identifier her
additional_datasets = []

microsoft_unlabeled_datasets = ["ms_middle-tier-api-dependency-latency-mv"]


@router.get("/start_evaluation")
async def start_evaluation(background_tasks: BackgroundTasks,
                           prom_con: dict = Depends(
                               connect_to_prometheus_instances),
                           db_session: session = Depends(load_session),
                           conf: dict = Depends(load_config)):
    """Handles the HTTP Route for /start_evaluation and delegates the function to the background queue

    :param background_tasks: The built-in background task queue from FastAPI
    :param prom_con: Dependence to the Prometheus connection objects
    :param db_session: Dependence to the database session object
    :param conf: Dependence to the config map

    :return: Returns text that the job was started as HTTP Response

    """

    background_tasks.add_task(evaluate_all, prom_con, db_session, conf)
    return "Started job: evaluation of all algorithms"


def evaluate_all(prom_con: dict, db_session: session, conf: dict):
    """Evaluates all configured anomaly detection frameworks and algorithms with the configured datasets on their
    performance and stores the results in the database

    :param prom_con: Prometheus connection objects
    :param db_session: Database session object
    :param conf: Config map

    """
    # c.inc()
    datasets = conf["evaluation"]["datasets"]
    evaluation_result = {}

    """Iterate over all datasets configured in config_map"""
    for dataset in datasets:
        dataset_id = dataset["id"]
        local_path = dataset["local_path"]
        dataset_labeled = dataset["labeled"]
        unsupervised = dataset["unsupervised"]
        ts_type = dataset["ts_type"]

        logging.info("Starting evaluation of dataset {0}".format(dataset_id))

        # Dataset can trained supervised when labeled, skip this dataset
        if not unsupervised and not dataset_labeled:
            continue

        """Load dataset"""
        pd_df = None
        if local_path is not None and local_path != "":
            if dataset_id in microsoft_datasets or dataset_id in microsoft_datasets_other_dt_format:
                pd_df = dask_workaround_read_single_csv_dir(local_path)
                pd_df.rename(columns={"TimeStamp": "timestamp", "Value": "value", "Label": "label"}, inplace=True)
            elif dataset_id in nab_datasets:
                pd_df = dask_workaround_read_single_csv_dir(local_path)
                # pd_df.rename(columns={"TimeStamp": "timestamp", "Value": "value"}, inplace=True)
            elif dataset_id in commercial_datasets:
                pd_df = dask_workaround_read_single_csv_dir(local_path)
            elif dataset_id in additional_datasets:
                # Extension of Datasets: add the dataset loading and the transformation of the datasets here
                # Extension of Datasets: remove the break statement
                break
            else:
                logging.info("Dataset not found - skip this dataset")
                break

        """Train the dataset with every specified algorithms"""
        for algorithm in dataset["algorithms"]:
            algo_id = algorithm["id"]
            train_percentage = algorithm["train_percentage"]
            identity = "evaluation_" + algo_id + "_with_dataset_" + dataset_id
            df_length = len(pd_df.index)

            df = pd_df.copy(deep=True)

            if dataset_id in microsoft_datasets:
                """Convert string to unix timestamp"""
                df["timestamp"] = df["timestamp"].apply(lambda x: datetime.strptime(x, DT_FORMAT).timestamp())
            elif dataset_id in microsoft_datasets_other_dt_format:
                """Convert string to unix timestamp"""
                df["timestamp"] = df["timestamp"].apply(lambda x: datetime.strptime(x, DT_FORMAT_ALTERNATIVE) \
                                                        .timestamp())
            elif dataset_id in nab_datasets:
                df["timestamp"] = df["timestamp"].apply(lambda x: datetime.strptime(x, DT_FORMAT_ALTERNATIVE).timestamp())
            elif dataset_id in commercial_datasets:
                df["timestamp"] = df["timestamp"].apply(lambda x: datetime.strptime(x, DT_FORMAT_ALTERNATIVE).timestamp())
            elif dataset_id in additional_datasets:
                # Extension of Datasets: add the dataset loading and the transformation of the dataset that is needed
                #   for the algorithm here
                # Extension of Datasets: remove the break statement
                break
            else:
                logging.info("Dataset not found - skip this dataset")
                break
            df.set_index("timestamp", inplace=True)

            """Split into train and test set according to config"""
            train_data_length = int(train_percentage * df_length)
            test_data_length = int(df_length - train_data_length)

            train_df = df.iloc[0:train_data_length]
            test_df = df.iloc[train_data_length:train_data_length + test_data_length]

            """Cache test_df for later comparison"""
            train_value_counts = None
            test_value_counts = None
            contamination_train = 0.1
            contamination_test = 0.1
            test_df_cache = test_df.copy(deep=True)

            """Training phase"""
            clf = None
            success = False

            if dataset_labeled and "contamination_train" not in algorithm:
                train_value_counts = train_df["label"].value_counts()
                test_value_counts = test_df["label"].value_counts()
                if 0 not in train_value_counts:
                    s = pd.Series(data={0: 0})
                    train_value_counts = train_value_counts.append(s)
                if 1 not in train_value_counts:
                    s = pd.Series(data={1: 0})
                    train_value_counts = train_value_counts.append(s)
                if 0 not in test_value_counts:
                    s = pd.Series(data={0: 0})
                    test_value_counts.append(s)
                if 1 not in test_value_counts:
                    s = pd.Series(data={1: 0})
                    test_value_counts.append(s)
                contamination_train = train_value_counts.get(1).item() \
                                      / (train_value_counts.get(0).item() + train_value_counts.get(1).item())
                # contamination_test = test_value_counts.get(1).item() \
                #                      / (test_value_counts.get(0).item() + test_value_counts.get(1).item())
            else:
                contamination_train = algorithm["contamination_train"]
                # contamination_test = contamination_train

            # Contamination Train can't be null, so it is set to a very small number, but training should not be done
            # on data without anomalies
            if contamination_train == 0.0:
                contamination_train = 0.00000001

            # Remove for training unsupervised
            if dataset_labeled and unsupervised:
                test_df = test_df.drop(columns="label")
                train_df = train_df.drop(columns="label")
            elif dataset_labeled and not unsupervised:
                test_df["label"] = 0
            logging.info("Starting training phase of {0}".format(algo_id))

            # Extension of Algorithms: Add the algorithm identifier (can be freely chosen, but must be unique) here and
            #   the corresponding algorithm class and training methods (that was added to the `algorithms` folder)

            if algo_id == "iforest":
                """iForest Training"""
                if unsupervised:
                    clf = IForestAlgorithm(contamination_train, ts_type, 1, 2)
                    clf.train_algorithm_unsupervised(train_df)
                else:
                    clf = IForestAlgorithm(contamination_train, ts_type, 2, 2)
                    clf.train_algorithm_supervised(train_df)
            elif algo_id == "cblof":
                if unsupervised:
                    clf = CBLOFAlgorithm(contamination_train)
                    clf.train_algorithm_unsupervised(train_df)
                else:
                    clf = CBLOFAlgorithm(contamination_train)
                    clf.train_algorithm_unsupervised(train_df)
            elif algo_id == "hbos":
                if unsupervised:
                    clf = HBOSAlgorithm(contamination_train)
                    clf.train_algorithm_unsupervised(train_df)
                else:
                    clf = HBOSAlgorithm(contamination_train)
                    clf.train_algorithm_supervised(train_df)
            elif algo_id == "knn":
                if unsupervised:
                    clf = KNNAlgorithm(contamination_train)
                    clf.train_algorithm_unsupervised(train_df)
                else:
                    clf = KNNAlgorithm(contamination_train)
                    clf.train_algorithm_supervised(train_df)
            elif algo_id == "oneclasssvm":
                if unsupervised:
                    clf = OneClassSVMAlgorithm(contamination_train)
                    clf.train_algorithm_unsupervised(train_df)
                else:
                    clf = OneClassSVMAlgorithm(contamination_train)
                    clf.train_algorithm_supervised(train_df)
            elif algo_id == "lmdd":
                if unsupervised:
                    clf = LMDDAlgorithm(contamination_train)
                    clf.train_algorithm_unsupervised(train_df)
                else:
                    clf = LMDDAlgorithm(contamination_train)
                    clf.train_algorithm_supervised(train_df)
            elif algo_id == "copod":
                if unsupervised:
                    clf = COPODAlgorithm(contamination_train)
                    clf.train_algorithm_unsupervised(train_df)
                else:
                    clf = COPODAlgorithm(contamination_train)
                    clf.train_algorithm_supervised(train_df)
            elif algo_id == "loda":
                if unsupervised:
                    clf = LODAAlgorithm(contamination_train)
                    clf.train_algorithm_unsupervised(train_df)
                else:
                    clf = LODAAlgorithm(contamination_train)
                    clf.train_algorithm_supervised(train_df)

            current_datetime = datetime.now().strftime(DT_FORMAT)

            if clf is not None:
                success = clf.store_model_to_file("temp/" + identity)

            if success:
                evaluation_result[identity] = True
            else:
                evaluation_result[identity] = False

            dataset_univariate = (ts_type == "univariate")

            """Evaluation phase"""
            logging.info("Starting evaluation phase of {0}".format(algo_id))
            algorithms = ["iforest", "copod", "loda", "cblof", "hbos", "knn", "oneclasssvm", "arima",
                          "autoencoder_torch", "abod",
                          "xgbod", "lmdd"]
            additional_algorithms = []
            if algo_id in algorithms:
                prediction, prediction_outlier_scores = clf.predict_sample(test_df, ts_type, unsupervised)
                result_df = prepare_and_store_dataframe(test_df_cache, current_datetime, prediction, identity,
                                                        conf["evaluation"]["df_output_dir"])
                build_comparison_graphs(result_df, current_datetime, identity, train_percentage, contamination_train,
                                        unsupervised, conf["evaluation"]["graph_output_dir"])
                eval_1_values, eval_2_values = analyze_performance(result_df, prediction_outlier_scores,
                                                                   dataset_labeled)
                persist_evaluation(db_session, dataset_labeled, current_datetime, dataset_id, local_path, dataset_univariate,
                                   dataset_labeled, contamination_train, unsupervised, algo_id, train_percentage, train_data_length,
                                   test_data_length, eval_1_values, eval_2_values)
            elif algo_id in additional_algorithms:
                # Extension of Algorithms: Add the evaluation code for your algorithm here
                # Extension of Algorithms: remove the break statement
                break
            elif algo_id == "prophet":
                print("lel")
                # prediction_df = clf.predict_sample(test_df, ts_type)
                # result_df = prepare_and_store_dataframe(prediction_df, current_datetime, prediction_df["anomaly"],
                #                                         identity, conf["evaluation"]["df_output_dir"])
                # result_df.rename(columns={"fact": "value", "anomaly": "label"}, inplace=True)
                # build_comparison_graphs(result_df, current_datetime, identity,
                #                         conf["evaluation"]["graph_output_dir"])
                # eval_1_values, eval_2_values = analyze_performance(result_df, None)
                # persist_evaluation(db_session, current_datetime, dataset_id, local_path, dataset_univariate, dataset_labeled, contamination_train, algo_id, train_percentage, train_data_length,
                #                    test_data_length, eval_1_values, eval_2_values)

            else:
                logging.info("Failed to evalute with algorithm {0}".format(algo_id))

            """Remove temporary evaluation model files"""
            try:
                os.remove("temp/" + identity + ".joblib")
            except FileNotFoundError as e:
                logging.info(e)
                logging.info("Failed to delete temporary file {0}, model could not be found".format("temp/" + identity
                                                                                                    + "joblib"))

    logging.info("Finished job: evaluation of all algorithms")
    return


def dask_workaround_read_csv_dir(path):
    path = path + "/*.csv"
    csv_files = glob.glob(path)
    dataframes = list()

    for f in csv_files:
        df = pd.read_csv(f)
        dataframes.append(df)

    result = pd.concat(dataframes)
    return result


def dask_workaround_read_single_csv_dir(path):
    f = open(path, "r")
    df = pd.read_csv(f)
    return df
