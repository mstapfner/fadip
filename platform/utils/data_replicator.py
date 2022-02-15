import copy
import sys
from datetime import datetime, timedelta

import pandas
import pandas as pd

DT_FORMAT = '%Y-%m-%dT%H:%M:%SZ'


def replicate_data(file_name: str, file_out: str):
    df = pandas.read_csv(file_name)
    end_date = df["TimeStamp"].iloc[-1]
    end_date = datetime.strptime(end_date, DT_FORMAT)
    step = datetime.strptime(df["TimeStamp"].iloc[1], DT_FORMAT) - datetime.strptime(df["TimeStamp"].iloc[0], DT_FORMAT)
    step_in_s = step.total_seconds()
    new_df = copy.deepcopy(df)
    new_dt = end_date
    for index, row in new_df.iterrows():
        new_dt = new_dt + timedelta(seconds=step_in_s)
        tm = new_dt.strftime(DT_FORMAT)
        new_df["TimeStamp"].iloc[index] = tm

    result = pd.concat([df, new_df], axis=0)
    result.reset_index(drop=True, inplace=True)
    result.to_csv(file_out, index=False)
    return


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        file_out = sys.argv[2]
        replicate_data(filename, file_out)
