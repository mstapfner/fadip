import logging
import os
import pathlib

from joblib import load


def download_and_save_model(name, bucket, client):
    """ Downloads the models from S3 and stores them in a temporary folder

    :param name: Name of the model file in S3
    :param bucket: Bucket to download from
    :param client: Boto3 client

    :return: the loaded model file
    """
    name = name + ".joblib"
    file_path = pathlib.Path(pathlib.Path.cwd(), "temp/", name)
    with open(file_path, 'wb') as f:
        client.download_fileobj(bucket, name, f)
    model = load(file_path)
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        logging.info("The model file does not exist, download and deletion failed")
        return False
    return model


