terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 3.27"
    }
  }

  required_version = ">= 0.14.9"
}

provider "aws" {
  profile = "default"
  region  = "eu-central-1"
}

resource "aws_cloudwatch_event_rule" "aws_cloudwatch_event_rule" {
  name                = "cloudwatch_event_rule_every_one_hour"
  description         = "Fires every hour"
  schedule_expression = "rate(1 hour)"
}

resource "aws_cloudwatch_event_target" "event_trigger_fadip_prediction" {
  rule      = aws_cloudwatch_event_rule.aws_cloudwatch_event_rule.name
  target_id = "lambda"
  arn       = aws_lambda_function.trigger_fadip_prediction.arn
}

resource "aws_lambda_permission" "allow_cloudwatch_to_call_trigger_fadip_prediction" {
  statement_id  = "AllowExecutionFromCloudWatch"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.trigger_fadip_prediction.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.aws_cloudwatch_event_rule.arn
}

resource "aws_iam_role" "iam_for_lambda_trigger_fadip_prediction" {
  name = "iam_for_lambda"

  assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Action": "sts:AssumeRole",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Effect": "Allow",
      "Sid": ""
    }
  ]
}
EOF
}

resource "aws_iam_role" "iam_role_anomaly_detection" {
  name = "iam_role_anomaly_detection"
  assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Action": [
        "sts:AssumeRole"
      ],
      "Effect": "Allow",
      "Principal": {
        "Service": [
          "ec2.amazonaws.com"
        ]
      }
    }
  ]
}
EOF
}

resource "aws_iam_instance_profile" "anomaly_detection_instance_profile" {
  name = "iam_anomaly_detection_instance_profile"
  role = ""
}

resource "aws_lambda_function" "trigger_fadip_prediction" {
  filename      = "lambda_function_payload.zip"
  function_name = "trigger_fadip_prediction"
  role          = aws_iam_role.iam_for_lambda_trigger_fadip_prediction.arn
  handler       = "main.lambda_handler"

  # The filebase64sha256() function is available in Terraform 0.11.12 and later
  # For Terraform 0.11.11 and earlier, use the base64sha256() function and the file() function:
  # source_code_hash = "${base64sha256(file("lambda_function_payload.zip"))}"
  source_code_hash = filebase64sha256("lambda_function_payload.zip")

  runtime = "python3.8"

  environment {
    variables = {
      FADIP_URL = "http://5.189.144.254"
    }
  }
}


resource "aws_instance" "anomaly_detection_instance" {
  ami = "ami-0a49b025fffbbdac6"
  instance_type = "t2.micro"
  iam_instance_profile = ""
  associate_public_ip_address = "false"


}