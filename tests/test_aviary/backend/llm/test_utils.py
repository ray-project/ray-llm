from unittest.mock import MagicMock, patch

from pytest import mark, raises

from rayllm.backend.llm.utils import get_aws_credentials, get_gcs_bucket_name_and_prefix
from rayllm.backend.server.models import S3AWSCredentials


@patch("os.getenv")
@patch("requests.post")
def test_get_aws_credentials_with_auth_token(mock_post, mock_getenv):
    mock_getenv.return_value = "dummy_token"
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "AWS_ACCESS_KEY_ID": "dummy_access_key",
        "AWS_SECRET_ACCESS_KEY": "dummy_secret_key",
    }
    mock_response.ok = True
    mock_post.return_value = mock_response

    credentials_config = S3AWSCredentials(
        auth_token_env_variable="TOKEN_ENV_VAR",
        create_aws_credentials_url="http://dummy-url.com",
    )
    result = get_aws_credentials(credentials_config)

    assert result == {
        "AWS_ACCESS_KEY_ID": "dummy_access_key",
        "AWS_SECRET_ACCESS_KEY": "dummy_secret_key",
    }
    mock_getenv.assert_called_once_with("TOKEN_ENV_VAR")
    mock_post.assert_called_once_with(
        "http://dummy-url.com", headers={"Authorization": "Bearer dummy_token"}
    )


@patch("requests.post")
def test_get_aws_credentials_without_auth_token(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "AWS_ACCESS_KEY_ID": "dummy_access_key",
        "AWS_SECRET_ACCESS_KEY": "dummy_secret_key",
    }
    mock_response.ok = True
    mock_post.return_value = mock_response

    credentials_config = S3AWSCredentials(
        auth_token_env_variable=None, create_aws_credentials_url="http://dummy-url.com"
    )
    result = get_aws_credentials(credentials_config)

    assert result == {
        "AWS_ACCESS_KEY_ID": "dummy_access_key",
        "AWS_SECRET_ACCESS_KEY": "dummy_secret_key",
    }
    mock_post.assert_called_once_with("http://dummy-url.com", headers=None)


@patch("requests.post")
def test_get_aws_credentials_request_failure(mock_post):
    mock_response = MagicMock()
    mock_response.ok = False
    mock_response.reason = "Bad Request"
    mock_post.return_value = mock_response

    credentials_config = S3AWSCredentials(
        auth_token_env_variable=None,
        create_aws_credentials_url="http://dummy-url.com",
    )
    result = get_aws_credentials(credentials_config)

    assert result is None
    mock_post.assert_called_once_with(
        "http://dummy-url.com",
        headers=None,
    )


class TestGetGcsBucketNameAndPrefix:
    def run_and_validate(
        self, gcs_uri: str, expected_bucket_name: str, expected_prefix: str
    ):
        bucket_name, prefix = get_gcs_bucket_name_and_prefix(gcs_uri)

        assert bucket_name == expected_bucket_name
        assert prefix == expected_prefix

    @mark.parametrize("trailing_slash", [True, False])
    def test_plain_bucket_name(self, trailing_slash: bool):
        gcs_uri = "gs://bucket_name"
        if trailing_slash:
            gcs_uri += "/"

        expected_bucket_name = "bucket_name"
        expected_prefix = ""

        self.run_and_validate(gcs_uri, expected_bucket_name, expected_prefix)

    @mark.parametrize("trailing_slash", [True, False])
    def test_bucket_name_with_prefix(self, trailing_slash: bool):
        gcs_uri = "gs://bucket_name/my/prefix"
        if trailing_slash:
            gcs_uri += "/"

        expected_bucket_name = "bucket_name"
        expected_prefix = "my/prefix/"

        self.run_and_validate(gcs_uri, expected_bucket_name, expected_prefix)

    def test_invalid_uri(self):
        gcs_uri = "s3://bucket/prefix"
        expected_bucket_name = None
        expected_prefix = None

        with raises(ValueError):
            self.run_and_validate(gcs_uri, expected_bucket_name, expected_prefix)
