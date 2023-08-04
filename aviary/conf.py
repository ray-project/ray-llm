import json
import logging
import os

import boto3

logger = logging.getLogger(__name__)

ENV_TOKEN_OVERRIDES = os.getenv("AVIARY_ENV_AWS_SECRET_NAME", "aviary/env_overrides")
ENV_VARS_TO_PROPAGATE = {"HUGGING_FACE_HUB_TOKEN"}


class SecretManager:
    def __init__(self, secret_name: str = ENV_TOKEN_OVERRIDES):
        self.secret_overrides = self.get_all_secrets(secret_name)

    def get_all_secrets(self, secret_name: str):
        try:
            aws_region_name = os.getenv("AWS_REGION", "us-west-2")

            # Create a Secrets Manager client
            session = boto3.session.Session()
            client = session.client(
                service_name="secretsmanager", region_name=aws_region_name
            )
            get_secret_value_response = client.get_secret_value(SecretId=secret_name)

            # Decrypts secret using the associated KMS key.
            secret = get_secret_value_response["SecretString"]

            secret_dict = json.loads(secret)
            return secret_dict
        except Exception as e:
            logger.debug(
                f"Unable to load env override secrets from {secret_name}. Using default secrets from env. {e}"
            )
            return {}

    def override_secret(self, env_var_name: str, set_in_env=True):
        # First read from env var, then from aws secrets
        secret = os.getenv(env_var_name, self.secret_overrides.get(env_var_name))
        if secret is None:
            logger.info(f"Secret {env_var_name} was not found.")
        elif set_in_env:
            os.environ[env_var_name] = secret
            logger.info(f"Secret {env_var_name} was set in the env.")
        return secret


secrets = SecretManager()
hugging_face_token = secrets.override_secret("HUGGING_FACE_HUB_TOKEN")
