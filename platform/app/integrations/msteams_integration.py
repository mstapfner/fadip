import logging

import pymsteams
from app.models.anomaly import Anomaly


def send_anomaly_to_msteams(anomaly: Anomaly, graph_path: str, webhook_url: str):

    """Send the detected anomaly object formatted to a Microsoft Teams channel

    :param anomaly: Anomaly object
    :param graph_path: Path to the created graph
    :param webhook_url: Webhook URL of Microsoft Teams channel

    :return: response object
    """

    anomaly_teams_msg = pymsteams.connectorcard(webhook_url)

    anomaly_teams_msg.title("Warning: Found anomaly")

    text_section = pymsteams.cardsection()

    anomaly_text = f"Found anomaly with query {anomaly.query_name} detected using {anomaly.detected_by}" + \
                           f". The first datetime of the anomaly is {anomaly.first_anomaly_datetime}, the last " \
                           f"datetime of the anomaly is {anomaly.last_anomaly_datetime}"

    text_section.text(anomaly_text)

    # TODO: Fix the image section
    # image_section = pymsteams.cardsection()
    # image_section.addImage()

    anomaly_teams_msg.addSection(text_section)
    anomaly_teams_msg.text(anomaly_text)
    # anomaly_teams_msg.addSection(image_section)
    try:
        response = anomaly_teams_msg.send()
    except Exception as e:
        logging.info("Failed to send notification to MS Teams Channel")
        logging.info(e)

    return response
