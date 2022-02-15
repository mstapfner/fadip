import glob
import sys

import pandas

DT_FORMAT = '%Y-%m-%dT%H:%M:%SZ'


def merge_to_multivariate(files_dir: str, file_out: str):

    all_csv_files = glob.glob(files_dir + "/*.csv")
    first = True
    base_df = []
    i = 1
    for filename in all_csv_files:
        if first:
            base_df = pandas.read_csv(filename)
            first = not first
            base_df.rename(columns={"Value": "value_1"}, inplace=True)
            # del base_df["Label"]
        else:
            df = pandas.read_csv(filename)
            column_name = "value_" + str(i)
            base_df[column_name] = df["Value"]
        i += 1

    base_df.reset_index(drop=True, inplace=True)
    base_df.to_csv(file_out, index=False)
    return


if __name__ == "__main__":
    if len(sys.argv) > 1:
        files_dir = sys.argv[1]
        file_out = sys.argv[2]
        merge_to_multivariate(files_dir, file_out)
