import pandas as pd
from dependencies.config_dependencies import load_config
from dependencies.database_dependencies import load_session
from fastapi import APIRouter, Depends
from matplotlib import pyplot as plt
from models.evaluation import Evaluation
from sqlalchemy import select
from sqlalchemy.orm import session

router = APIRouter()

colors = {
    "iforest": "tab:blue",
    "knn": "tab:orange",
    "hbos": "tab:brown",
    "copod": "tab:cyan",
    "cblof": "tab:olive"
}

line_styles = {
    "iforest": "o-.",
    "knn": "+-.",
    "hbos": "x--",
    "copod": "-.",
    "cblof": "<:"
}

@router.get("/stat_routes")
async def stat_routes(db_session: session = Depends(load_session),
                           conf: dict = Depends(load_config)):

    dataset_map = {}

    # For every dataset
    datasets = conf["evaluation"]["datasets"]
    for dataset in datasets:
        algorithm_data = {}
        dataset_id = dataset["id"]

        for algorithm in dataset["algorithms"]:
            # Request the mcc score, f1 score etc.
            algo_id = algorithm["id"]
            train_percentage = algorithm["train_percentage"]
            identity = "evaluation_" + algo_id + "_with_dataset_" + dataset_id
            evaluations = db_session.query(Evaluation).where(Evaluation.dataset_id == dataset_id,
                                                            Evaluation.evaluated_by == algo_id).all()
            train_test_split_map = {}
            for eval in evaluations:
                train_test_split_map[eval.train_percentage] = eval
            algorithm_data[algo_id] = train_test_split_map
        dataset_map[dataset_id] = algorithm_data

    metrics = ["mcc", "accuracy", "f1", "precision", "specificity"]

    for metric in metrics:
        for dataset in dataset_map:
            plt.rcParams["figure.figsize"] = (6, 4)
            plt.rcParams['axes.labelsize'] = 14
            plt.rcParams['axes.titlesize'] = 20
            plt.rcParams["font.size"] = 12
            plt.figure()


            x = []
            # df_map = {"train_test_split": [0.4, 0.5, 0.6, 0.7]}
            df_map = {"train_test_split": [0.6, 0.7, 0.8, 0.9]}
            df = pd.DataFrame(df_map)

            for algorithm in dataset_map[dataset]:
                column_name = algorithm + "_" + metric
                mcc_scores = {}
                for train_test_split in dataset_map[dataset][algorithm]:
                    if metric == "mcc":
                        mcc_scores[train_test_split] = dataset_map[dataset][algorithm][train_test_split].mcc
                    elif metric == "accuracy":
                        mcc_scores[train_test_split] = dataset_map[dataset][algorithm][train_test_split].accuracy
                    elif metric == "f1":
                        mcc_scores[train_test_split] = dataset_map[dataset][algorithm][train_test_split].f1_score
                    elif metric == "precision":
                        mcc_scores[train_test_split] = dataset_map[dataset][algorithm][train_test_split].precision
                    elif metric == "specificity":
                        mcc_scores[train_test_split] = dataset_map[dataset][algorithm][train_test_split].specificity
                df[column_name] = df["train_test_split"].map(mcc_scores)

            columns = df.columns
            for column in columns:
                if not column == "train_test_split":
                    if metric in column:
                        cache = column.split("_")
                        line_style = line_styles.get(cache[0])
                        color = colors.get(cache[0])
                        plt.plot(df["train_test_split"], df[column], "" + line_style, color=color,
                                 label=cache[0])

            plt.xlabel("Train-Test-Split: Train %")
            label = ""
            if metric == "mcc":
                label = "MCC Score"
            elif metric == "accuracy":
                label = "Accuracy Score"
            elif metric == "f1":
                label = "F1 Score"
            elif metric == "precision":
                label = "Precision Score"
            elif metric == "specificity":
                label = "Specificity Score"
            plt.ylabel(label)
            plt.tick_params(axis='y')
            plt.tight_layout()
            plt.grid(axis="both",
                     color="0.9",
                     linestyle='-',
                     linewidth=1)
            plt.legend()

            # if metric == "mcc" and dataset == "ms_data-ingress-rate-05":
            #     filename = "../evaluations/statistics/" + metric + "/" + dataset + ".pdf"
            #     plt.savefig(filename)

            filename = "../evaluations/statistics/" + metric + "/" + dataset + ".pdf"
            plt.savefig(filename)
    return "Success"