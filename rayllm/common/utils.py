import os
import traceback
import warnings

import boto3

try:
    from langchain.llms import OpenAIChat

    LANGCHAIN_INSTALLED = True
    LANGCHAIN_SUPPORTED_PROVIDERS = {"openai": OpenAIChat}
except ImportError:
    LANGCHAIN_INSTALLED = False


class ResponseError(RuntimeError):
    def __init__(self, *args: object, **kwargs) -> None:
        self.response = kwargs.pop("response", None)
        super().__init__(*args)


def _is_aviary_model(model: str) -> bool:
    """
    Determine if this is an aviary model. Aviary
    models do not have a '://' in them.
    """
    return "://" not in model


def _supports_batching(model: str) -> bool:
    provider, _ = model.split("://", 1)
    return provider != "openai"


def _get_langchain_model(model: str):
    if not LANGCHAIN_INSTALLED:
        raise ValueError(
            f"Unsupported model {model}. If you want to use a langchain-"
            "compatible model, install langchain ( pip install langchain )."
        )

    provider, model_name = model.split("://", 1)
    if provider not in LANGCHAIN_SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown model provider for {model}. Supported providers are: "
            f"{' '.join(LANGCHAIN_SUPPORTED_PROVIDERS.keys())}"
        )
    return LANGCHAIN_SUPPORTED_PROVIDERS[provider](model_name=model_name)


def _convert_to_aviary_format(model: str, llm_result):
    generation = llm_result.generations
    result_list = [{"generated_text": x.text} for x in generation[0]]
    return result_list


def has_ray():
    try:
        import ray  # noqa: F401

        return True
    except ImportError:
        warnings.warn(traceback.format_exc(), stacklevel=2)
        return False


def has_backend():
    try:
        import rayllm.backend  # noqa: F401

        return True
    except ImportError:
        warnings.warn(traceback.format_exc(), stacklevel=2)
        return False


def assert_has_ray():
    assert has_ray(), (
        "This command requires ray to be installed. "
        "Please install ray with `pip install 'ray[default]'`"
    )


def assert_has_backend():
    # TODO: aviary is not on pypi yet, instruction not actionable
    assert has_backend(), (
        "This command requires aviary backend to be installed. "
        "Please install backend dependencies with `pip install aviary[backend]`. "
    )


def download_files_from_s3(bucket_uri: str, dest_dir: str):
    """Download files from s3 to a local directory"""
    isExist = os.path.exists(dest_dir)
    if not isExist:
        os.makedirs(dest_dir)
    s3 = boto3.resource("s3")
    bucket_name, prefix = bucket_uri.replace("s3://", "").split("/", 1)
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=prefix):
        s3.meta.client.download_file(
            bucket_name, obj.key, f"{dest_dir}/{obj.key.split('/')[-1]}"
        )
