from app.algorithms.algorithm_meta import Algorithm_Meta
from pyod.models import copod
from joblib import dump, load
from app.utils.model_download_s3 import download_and_save_model
from app.utils.model_upload_s3 import store_and_upload_model


class COPODAlgorithm(metaclass=Algorithm_Meta):
    id = "copod"
    clf: copod.COPOD = None

    def __init__(self, contamination):
        self.clf = copod.COPOD(contamination=contamination)

    def train_algorithm_unsupervised(self, data):
        self.clf.fit(data)

    def train_algorithm_supervised(self, df):
        self.clf.fit(df)

    def predict_sample(self, dataframe, ts_type, unsupervised, dataset="standard"):
        if ts_type == "univariate" and unsupervised and dataset == "standard":
            dataframe = dataframe.values.reshape(-1, 1)
        prediction = self.clf.predict(dataframe)
        prediction_outlier_scores = self.clf.decision_function(dataframe)
        return prediction, prediction_outlier_scores

    def store_model_to_file(self, name):
        success = dump(self.clf, name + ".joblib")
        return success

    def store_model_to_s3(self, name, bucket, client):
        return store_and_upload_model(self.clf, name, bucket, client)

    def load_model_from_file(self, name):
        self.clf = load(name + ".joblib")

    def load_model_from_s3(self, name, bucket, client):
        self.clf = download_and_save_model(name, bucket, client)

