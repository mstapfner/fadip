import logging
from datetime import datetime
from typing import Optional

from prometheus_pandas import query
from utils.multivariation_detector import detect_prometheus_multivariation


def prom_custom_query(prom_con: query.Prometheus, qry: str, start_time: datetime, end_time: datetime, ts_type: str,
                      step: Optional[int]):

    """Fetch the datapoints of a query inside a specified time interval with configurable datapoint resolution

    :param prom_con: Configured prometheus connection object
    :param qry: PromQL query
    :param start_time: Timestamp at which the datapoints of the query start
    :param end_time: Timestamp at which the datapoints of the query end
    :param ts_type: Type of timeserie, either univariate or multivariate
    :param step: (Optional) Datapoint resolution, currently ignored

    :return: Datapoints as Pandas dataframe
    """

    metric_data = prom_con.query_range(query=qry, start=start_time, end=end_time, step='15s')

    if len(metric_data.index) <= 0:
        logging.info("Request for query: {0} in timeinterval {1} - {2} returned no result".format(qry, start_time,
                                                                                                  end_time))
        return []

    metric_data["timestamp"] = metric_data.index

    columns = metric_data.columns.tolist()
    amount_mv = len(columns)

    if ts_type == "univariate":
        # Rename columns to timestamp and value
        mapping = {metric_data.columns.tolist()[0]: "value"}
        metric_data.rename(columns=mapping, inplace=True)

    # Rebuild index
    metric_data.reset_index(drop=True, inplace=True)
    metric_data.set_index("timestamp", inplace=True, drop=True)

    return metric_data

