from app.algorithms.algorithm_meta import Algorithm_Meta
from pyod.models import iforest
from joblib import dump, load
from app.utils.model_download_s3 import download_and_save_model
from app.utils.model_upload_s3 import store_and_upload_model


class IForestAlgorithm(metaclass=Algorithm_Meta):
    id = "iforest"
    clf: iforest.IForest = None

    def __init__(self, contamination, ts_type, features, n_jobs):
        self.clf = iforest.IForest(contamination=contamination, max_features=features, n_jobs=n_jobs, behaviour="new")

    def train_algorithm_unsupervised(self, df):
        self.clf.fit(df)

    def train_algorithm_supervised(self, df):
        self.clf.fit(df)

    def predict_sample(self, df, ts_type, unsupervised):
        prediction = self.clf.predict(df)
        prediction_outlier_scores = self.clf.decision_function(df)
        return prediction, prediction_outlier_scores

    def store_model_to_file(self, name):
        success = dump(self.clf, name + '.joblib')
        return success

    def store_model_to_s3(self, name, bucket, client):
        return store_and_upload_model(self.clf, name, bucket, client)

    def load_model_from_file(self, name):
        self.clf = load(name + '.joblib')

    def load_model_from_s3(self, name, bucket, client):
        self.clf = download_and_save_model(name, bucket, client)
