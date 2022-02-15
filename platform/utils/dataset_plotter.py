import glob
import sys

import matplotlib.pyplot as plt
import pandas
from datetime import datetime

from matplotlib.dates import DateFormatter, DayLocator
import matplotlib.dates as mdates

DT_FORMAT = '%Y-%m-%dT%H:%M:%SZ'
DT_FORMAT_ALTERNATIVE = '%Y-%m-%d %H:%M:%S'


def plot_csv_file(file_name: str, file_out: str):
    dp = 600

    df = pandas.read_csv(file_name)
    # df["TimeStamp"] = df["TimeStamp"].apply(lambda x: datetime.strptime(x, DT_FORMAT))
    df["timestamp"] = df["timestamp"].apply(lambda x: datetime.strptime(x, DT_FORMAT_ALTERNATIVE))

    color = 'tab:blue'
    plt.rcParams["figure.figsize"] = (20, 5)
    # plt.plot(df["TimeStamp"], df["Value"], color=color)
    plt.plot(df["timestamp"], df["value"], color=color)
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.tick_params(axis='y', labelcolor=color)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.tight_layout()

    plt.savefig(file_out + '.pdf', bbox_inches='tight')
    return


def plot_csv_file_with_labels(file_name: str, file_out: str):
    dp = 600

    df = pandas.read_csv(file_name, converters={'TimeStamp': lambda x: str(x)})
    df["TimeStamp"] = df["TimeStamp"].apply(lambda x: datetime.strptime(x, DT_FORMAT_ALTERNATIVE))

    plt.rcParams["figure.figsize"] = (20, 5)

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_ylabel('Anomaly', color=color)  # we already handled the x-label with ax1
    ax1.plot(df["TimeStamp"], df["Label"], color=color, marker=".", alpha=0.35, zorder=10)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel('Value', color=color)
    ax2.plot(df["TimeStamp"], df["Value"], color=color, zorder=5)
    ax2.tick_params(axis='y', labelcolor=color)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(file_out + '.pdf', bbox_inches='tight')
    return


if __name__ == "__main__":
    # Args: CSV-File, Output Filename, Boolean whether labeled or not labeled
    if len(sys.argv) > 2:
        if sys.argv[3] == "False" or sys.argv[3] == "false" or sys.argv[3] is False:
            plot_csv_file(file_name=sys.argv[1], file_out=sys.argv[2])
        else:
            plot_csv_file_with_labels(file_name=sys.argv[1], file_out=sys.argv[2])
