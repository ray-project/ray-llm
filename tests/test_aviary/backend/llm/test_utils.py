from unittest.mock import MagicMock, patch

from aviary.backend.llm.utils import get_aws_credentials
from aviary.backend.server.models import S3AWSCredentials


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
