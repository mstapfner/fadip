from app.algorithms.algorithm_meta import Algorithm_Meta
from pyod.models import auto_encoder_torch
from joblib import dump, load
from app.utils.model_download_s3 import download_and_save_model
from app.utils.model_upload_s3 import store_and_upload_model


class AutoEncoderAlgorithm(metaclass=Algorithm_Meta):
    id = "autoencoder_torch"
    clf: auto_encoder_torch.AutoEncoder = None

    def __init__(self, contamination):
        self.clf = auto_encoder_torch.AutoEncoder(contamination=contamination)

    def train_algorithm_unsupervised(self, data, ts_type):
        if ts_type == "univariate":
            data = data.values.reshape(-1, 1)
        self.clf.fit(data)

    def predict_sample(self, dataframe, ts_type, unsupervised):
        if ts_type == "univariate" and unsupervised:
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

