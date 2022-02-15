import logging

from app.models import anomaly
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


def send_anomaly_to_slack(anomaly: anomaly.Anomaly, graph_path: str, oauth_token: str):

    """Send the detected anomaly object formatted to a Slack channel

    :param anomaly: Anomaly object
    :param graph_path: Path to the created graph
    :param oauth_token: OAuth Token for authentication to the Slack channel

    """

    client = WebClient(token=oauth_token)
    try:
        # Post message
        response = client.chat_postMessage(
            channel='#anomalien',
            text=f"Warning: Found anomaly with the query {anomaly.query_name} detected using {anomaly.detected_by}."
                 f"The first datetime of the anomaly is {anomaly.first_anomaly_datetime}, the last datetime of anomaly "
                 f"is {anomaly.last_anomaly_datetime}",
            blocks=[
                {
                    "type": "divider"
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":warning: Found anomaly: :warning:"
                                f"\n\n*Query:* `{anomaly.query_name}`" 
                                f"\n*Algorithm:* `{anomaly.detected_by}`"
                                f"\n\n*First anomaly datetime:* {anomaly.first_anomaly_datetime}"
                                f"\n*Last anomaly datetime:* {anomaly.last_anomaly_datetime}"
                                f"\n\n"
                    }
                }
            ]
        )
        file_response = client.files_upload(channels="#anomalien",
                                            file=graph_path)

    except SlackApiError as e:
        logging.info("Failed to send notification to Slack")
        assert e.response["ok"] is False
        assert e.response["error"]
        logging.info(f"Got an error: {e.response['error']}")

    return
