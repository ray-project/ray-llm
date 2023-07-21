import pytest

from aviary.common.constants import MODELS
from aviary.sdk import models


# TODO (max) fix checking for backend models == frontend models
# doesn't work, as we only run a single test model in the backend for testing.
@pytest.mark.skip()
def test_model_descriptions_complete():
    all_backend_models = models()
    all_frontend_models = MODELS.keys()

    print(all_backend_models)

    # TODO: Fix this test
    assert len(all_backend_models) <= len(all_frontend_models)
    # assert sorted(all_backend_models) == sorted(all_frontend_models)
