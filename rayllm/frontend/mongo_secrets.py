# Use this code snippet in your app.
# If you need more information about configurations
# or implementing the sample code, visit the AWS docs:
# https://aws.amazon.com/developer/language/python/

import json
import logging
import os

import boto3


def get_mongo_secret_url():
    mongo_url = os.getenv("MONGODB_URL")
    if mongo_url:
        return mongo_url
    try:
        secret_name = "prod/frontend/mongo_password"
        region_name = "us-west-2"

        # Create a Secrets Manager client
        session = boto3.session.Session()
        client = session.client(service_name="secretsmanager", region_name=region_name)

        get_secret_value_response = client.get_secret_value(SecretId=secret_name)

        # Decrypts secret using the associated KMS key.
        secret = get_secret_value_response["SecretString"]

        secret_dict = json.loads(secret)
        mongo_url = secret_dict.get("url")
        return mongo_url
    except Exception as e:
        # Fail quietly if we can't get the secret
        logging.warning(f"Failed to retrieve mongo secret, Exception: {e}")
