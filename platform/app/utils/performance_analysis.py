import math

import numpy
import pandas
from app.models.evaluation import Evaluation
from sqlalchemy.orm import session


def prepare_and_store_dataframe(test_df: pandas.DataFrame, current_datetime: str, prediction: numpy.ndarray,
                                eval_identity: str, df_output_dir: str):
    """Prepares a dataframe that includes the testing data (timestamp, value), the detected anomalies and the labeled
    anomalies from the dataset and stores this as a .pkl file on the disk

    :param test_df: Dataframe that includes the used testing data
    :param current_datetime: Current datetime as string to be included in filename
    :param prediction: The predicted anomalies as numpy array
    :param eval_identity: The evaluation identity, consists of the dataset and the used algorithm, is used in filename
    :param df_output_dir The output directory for resulting pickled dataframe

    """

    df = test_df.copy(deep=True)
    df["prediction"] = prediction
    df["timestamp"] = df.index

    """Pickle Dataframe and store to file"""
    df.to_pickle(df_output_dir + eval_identity + "-" + current_datetime + ".pkl")
    return df


def analyze_performance(result_df: pandas.DataFrame, prediction_outlier_scores: numpy.ndarray, dataset_labeled: bool):
    """Analyzes the performance of the anomaly detection by calculating true-positives, false-positives, true-negatives,
    false-negatives

    :param result_df: Dataframe that includes testing data, prediction and original anomaly labels
    :param prediction_outlier_scores: The exact prediction outlier scores of the algorithm
    :param dataset_labeled: True if used dataset was labeled

    """
    result = result_df.copy(deep=True)
    result["prediction_outlier_scores"] = prediction_outlier_scores

    if dataset_labeled:
        """eval_1 has value 2 for true positives, and value 0 for true negatives"""
        result["eval_1"] = result["label"] + result["prediction"]
        """eval_2 has value -1 for false positives and value 1 for false negatives"""
        result["eval_2"] = result["label"] - result["prediction"]
        eval_1_values = result["eval_1"].value_counts()
        eval_2_values = result["eval_2"].value_counts()
    else:
        eval_1_values = pandas.Series([0, 0])
        eval_2_values = pandas.Series([0, 0])

    return eval_1_values, eval_2_values


def persist_evaluation(db_session: session, labeled: bool, current_datetime: str, dataset_id: str, dataset_location: str,
                       dataset_univariate: bool, dataset_labeled: bool, contamination_train: float, unsupervised: bool,
                       evaluated_by: str, train_percentage: float, train_samples: int, test_samples: int,
                       eval_1_values: pandas.Series, eval_2_values: pandas.Series):
    """Persist the evaluation with all parameters to the PostgresDB

    :param db_session: Database session
    :param labeled: Whether the dataset is labeled or not
    :param current_datetime: Current datetime as string
    :param dataset_id: Dataset identifier
    :param dataset_location: Filepath of the dataset location
    :param dataset_univariate: True if dataset is univariate timeseries, false if multivariate
    :param dataset_labeled: True if the dataset is labeled
    :param contamination_train: If dataset is unlabeled, this contamination percentage was used for training
    :param unsupervised: If the training phase was unsupervised
    :param evaluated_by: Name of the used anomaly detection algorithm
    :param train_percentage: Percentage of data used for training
    :param train_samples: Amount of samples used in training dataset
    :param test_samples: Amount of samples used in test dataset
    :param eval_1_values: Series that contains the true positives and the true negatives
    :param eval_2_values: Series that contains the false positives and the false negatives

    """

    if dataset_labeled:
        true_positives = eval_1_values.get(2).item() if eval_1_values.get(2) is not None else 0
        true_negatives = eval_1_values.get(0).item() if eval_1_values.get(0) is not None else 0
        false_positives = eval_2_values.get(-1).item() if eval_2_values.get(-1) is not None else 0
        false_negatives = eval_2_values.get(1).item() if eval_2_values.get(1) is not None else 0

        # Advanced metrics
        accuracy = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)

        # Set precision and recall to very small numbers if they would be 0, so that f1 score can still be computed
        if (true_positives + false_positives) == 0:
            precision = 0.000000001
        else:
            precision = true_positives / (true_positives + false_positives)

        if (true_positives + false_negatives) == 0:
            recall = 0.000000001
        else:
            recall = true_positives / (true_positives + false_negatives)

        if (true_negatives + false_positives) == 0:
            specificity = 0.000000001
        else:
            specificity = true_negatives / (true_negatives + false_positives)

        if precision == 0:
            precision = 0.000000001
        if recall == 0:
            recall = 0.000000001
        f1_score = 2 / ((1 / precision) + (1 / recall))

        # float division by zero catcher
        mcc_below = (math.sqrt(
                  (true_positives + false_positives) *
                  (true_positives + false_negatives) *
                  (true_negatives + false_positives) *
                  (true_negatives + false_negatives)
              ))
        if mcc_below == 0:
            mcc_below = 0.000000001

        mcc = ((true_positives * true_negatives) + (false_positives * false_negatives)) / mcc_below
    else:
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        precision = 0
        recall = 0
        accuracy = 0
        f1_score = 0
        specificity = 0
        mcc = 0

    evaluation = Evaluation(
        query_name="",
        dataset_id=dataset_id,
        dataset_location=dataset_location,
        dataset_univariate=dataset_univariate,
        dataset_labeled=dataset_labeled,
        contamination_train=contamination_train,
        evaluated_by=evaluated_by,
        evaluation_date=current_datetime,
        train_percentage=train_percentage,
        unsupervised=unsupervised,
        train_samples=train_samples,
        test_samples=test_samples,
        true_positives=true_positives,
        true_negatives=true_negatives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1_score=f1_score,
        mcc=mcc,
    )

    result = db_session.add(evaluation)
    commit_result = db_session.commit()
    return
