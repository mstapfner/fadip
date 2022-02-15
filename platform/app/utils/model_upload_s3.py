import logging
import pathlib

import boto3
from botocore.exceptions import ClientError
from joblib import dump
import os


def store_and_upload_model(model, name, bucket, client):
    """ Stores the model as a temporary file on the local drive and uploads the model file to an S3 Bucket

    :param model: Model that will be stored and uploaded
    :param name: Name of the model file
    :param bucket: Bucket to upload to
    :param client: Boto3 client

    :return: True if model storage and upload was successful
    """

    file_name = name + ".joblib"
    file_path = pathlib.Path(pathlib.Path.cwd(), "temp/")
    if not file_path.exists():
        os.mkdir(file_path)
    file_path = pathlib.Path(pathlib.Path.cwd(), "temp/", file_name)
    dump(model, file_path)
    file_name = str(file_path.resolve())
    success = upload_file(client=client, file_name=file_name, bucket=bucket)
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        logging.info("The model file does not exist, upload and deletion failed")
        return False
    return success


# Adapted from: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-uploading-files.html
def upload_file(client, file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param client: Boto3 client
    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    try:
        response = client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True
