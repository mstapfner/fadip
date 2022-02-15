from datetime import datetime

import pandas
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from matplotlib.offsetbox import TextArea, VPacker, AnchoredOffsetbox
from pandas import DataFrame


def build_normal_data_graph(df: pandas.DataFrame, current_datetime: str, eval_id: str,
                            graph_output_dir: str):
    dp = 400
    plt.rcParams["figure.figsize"] = (18, 10)
    fig = plt.plot()

    return


def build_anomaly_graph_alerting(query_name: str, algorithm_name: str, df: DataFrame, identity: str, ts_type: str):

    """Builds a graph with the detected anomalies

    :param query_name: Prometheus Query of the affected timeseries
    :param algorithm_name: Anomaly detection algorithm that detected the anomalies
    :param df: Dataframe that includes the datapoints and the classification of them
    :param identity: Unique identifier for the anomalies
    :param ts_type: Type of timeseries, either "univariate" or "multivariate"
    :return: Local filepath of the graphs for further processing / sending
    """

    dp = 400
    plt.rcParams["figure.figsize"] = (18, 10)
    columns = df.columns.tolist()
    amount_features = len(columns) - 2
    axes = range(amount_features)

    fig, axes = plt.subplots(amount_features + 2, 1)
    plt.rcParams.update({'font.size': 6})
    fig.suptitle("Query: " + query_name + "\n predicted with: " + algorithm_name)

    df["time_cache"] = df["time_cache"].apply(lambda x: int(x))
    df["time_cache"] = df["time_cache"].apply(lambda x: datetime.fromtimestamp(x))

    columns.remove("anomaly")
    columns.remove("time_cache")

    i = 0
    for ax in axes:
        if i == 0:
            ax.plot(df["time_cache"], df["anomaly"], ".-r")
            ax.set_xlabel("timestamp")
            ax.set_ylabel('anomalies')
        elif i == 1:
            ax.plot(df["time_cache"], df["anomaly"], "-g")
            ax.set_xlabel("timestamp")
            ax.set_ylabel("anomaly probability")
        else:
            column_name = columns[i-2]
            ax.plot(df["time_cache"], df[column_name], "-b")
            ax.set_xlabel("timestamp")
            ax.set_ylabel(column_name)
        i += 1

    # store graph as file in cache
    file_path = "cache/" + identity + ".png"
    result = fig.savefig("cache/" + identity + ".png")

    return file_path


def build_comparison_graphs(df: pandas.DataFrame, current_datetime: str, eval_identity: str, train_test_split: float,
                            contamination_train: float, unsupervised: bool, graph_output_dir: str):
    """Creates a graph with two subplots out of the evaluation result dataframe, and stores them as png in the
    specified output folder

    :param df: Dataframe that includes testing data, prediction and original anomaly labels
    :param current_datetime: Current datetime as string to be included in filename
    :param eval_identity: The evaluation identity, consists of the dataset and the used algorithm, is used in filename
    :param train_test_split: The train-test-split used for this evaluation
    :param contamination_train: The contamination used for this evaluation
    :param unsupervised: Whether the training phase was unsupervised or not
    :param graph_output_dir: The output directory for resulting graphs

    """

    dp = 400
    plt.rcParams["figure.figsize"] = (18, 10)
    fig = None

    df["timestamp"] = df["timestamp"].apply(lambda x: datetime.utcfromtimestamp(x))

    columns = df.columns.tolist()
    amount_features = len(columns) - 3
    axes = range(amount_features)

    columns.remove("timestamp")
    if "label" in df:
        columns.remove("label")
    else:
        amount_features += 1
    columns.remove("prediction")

    color_pred = 'tab:red'
    color_lel = 'tab:orange'
    color = 'tab:blue'
    color_label = 'tab:green'

    flagger = True
    if flagger:
        if "reg" in eval_identity and "app" in eval_identity:
            if train_test_split <= 0.8:
                plt.rcParams["figure.figsize"] = (14, 2.8)
            else:
                plt.rcParams["figure.figsize"] = (5, 4)
        else:
            plt.rcParams["figure.figsize"] = (20, 5)
        fig, ax1 = plt.subplots()
        if "label" in df:
            df["eval_1"] = df["prediction"] + df["label"] # ist 2, weil pred = 1, label = 1
            df["eval_2"] = df["label"] - df["prediction"]

            df["tp"] = df["eval_1"].apply(lambda x: 1 if (x == 2) else 0)
            df["fp"] = df["eval_2"].apply(lambda x: 1 if (x == -1) else 0)
            df["fn"] = df["eval_2"].apply(lambda x: 1 if (x == 1) else 0)

            ax1.plot(df["timestamp"], df["tp"], color=color_label, alpha=0.5, zorder=10, label='True positives')
            ax1.fill_between(df["timestamp"], df["tp"], color=color_label, alpha=0.3)
            ax1.plot(df["timestamp"], df["fp"], color=color_pred, alpha=0.4, zorder=10, label='False positives')
            ax1.fill_between(df["timestamp"], df["fp"], color=color_pred, alpha=0.25)
            ax1.plot(df["timestamp"], df["fn"], color=color_lel, alpha=0.5, zorder=10, label='False negatives')
            ax1.fill_between(df["timestamp"], df["fn"], color=color_lel, alpha=0.3)
            ax1.set_xlabel("timestamp")
            ax1.tick_params(axis='y')

            ax2 = ax1.twinx()
            ax2.set_xlabel('Timestamp')
            ax2.set_ylabel('Value', color=color)
            ax2.plot(df["timestamp"], df["value"], marker=".", color=color, zorder=5)
            ax2.tick_params(axis='y', labelcolor=color)

        else:
            ax1.plot(df["timestamp"], df["value"], color=color, zorder=10)
            ax1.set_ylabel("Value", color=color_pred)
            ax1.set_xlabel("Timestamp")
            ax1.fill_between(df["timestamp"], df["value"], color=color, alpha=0.3)

            ax2 = ax1.twinx()
            ax2.plot(df["timestamp"], df["prediction"], marker=".", color=color_pred,  alpha=0.5, zorder=5, label="Detected Anomalies")
            ax2.tick_params(axis='y', labelcolor=color)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

        if "reg" in eval_identity and "app" in eval_identity:
            if train_test_split == 0.98:
                fig.autofmt_xdate()
        fig.tight_layout()
        fig.legend()
        # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    else:
        if amount_features == 1:
            if "reg" in eval_identity and "app" in eval_identity:
                plt.rcParams["figure.figsize"] = (5, 4)
            else:
                plt.rcParams["figure.figsize"] = (20, 5)

            fig, ax1 = plt.subplots()
            ax1.set_ylabel("Anomaly", color=color_pred)
            ax1.plot(df["timestamp"], df["label"], color=color_label, alpha=0.95, marker=".", zorder=10)
            ax1.plot(df["timestamp"], df["prediction"], color=color_pred, alpha=0.45, marker="o", zorder=10)
            ax1.tick_params(axis='y', labelcolor=color_label)

            ax2 = ax1.twinx()
            ax2.set_xlabel('Timestamp')
            ax2.set_ylabel('Value', color=color)
            ax2.plot(df["timestamp"], df["value"], color=color, alpha=1, zorder=5)
            ax2.tick_params(axis='y', labelcolor=color)

            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

            fig.tight_layout()  # otherwise the right y-label is slightly clipped

        else:
            fig, axes = plt.subplots(amount_features + 1, 1)
            i = 0
            for ax in axes:
                if i == 0:
                    if "label" in df:
                        ax1 = ax.twinx()
                        ax.plot(df["timestamp"], df["prediction"], color=color_pred, marker=".", zorder=10)
                        ax1.plot(df["timestamp"], df["label"], color=color_label, marker=".", zorder=5)
                        # ax1.plot(df["timestamp"], df["value"], color=color)
                        ax.set_xlabel("Timestamp")
                        ax.set_ylabel("Predicted Anomalies", color="r")
                        ax1.set_ylabel("Actual Anomalies", color="g")
                    else:
                        ax.plot(df["timestamp"], df["prediction"], color=color_pred, marker=".", alpha=0.34)
                        ax.set_xlabel("timestamp")
                        ax.set_ylabel("anomalies", color="r")
                else:
                    column_name = columns[i-2]
                    ax.plot(df["timestamp"], df[column_name], color=color)
                    ax.set_xlabel("timestamp")
                    ax.set_ylabel(column_name, color="b")
                i += 1

    us_flag = "-supervised-"
    if unsupervised:
        us_flag = "-unsupervised-"
    fig.savefig(graph_output_dir + eval_identity + "-tts_" + str(train_test_split) + us_flag + "-conttrain" +
                str(contamination_train) + "-" + current_datetime + '.pdf')
    plt.cla()
    plt.clf()
    return
