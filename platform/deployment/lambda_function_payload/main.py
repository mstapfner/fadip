import os
import json
import urllib3


def lambda_handler(event, context):
    url = os.environ["FADIP_URL"]
    url = url + "/predict_all"
    http = urllib3.PoolManager()
    r = http.request("GET", url)

    data = r.data

    return {
        "statusCode": r.status,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps(data)
    }