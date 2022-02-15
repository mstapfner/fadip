import boto3
from app.dependencies.config_dependencies import load_config


def load_aws_s3_client():

    """Create and load the aws s3 client

    :return: Returns the connected aws s3 client

    """

    config = load_config()

    # Connect to AWS S3
    client = boto3.client(
        's3',
        aws_access_key_id=config["model_storage"]["s3"]["aws_access_key_id"],
        aws_secret_access_key=config["model_storage"]["s3"]["aws_secret_access_key"]
    )

    return client
