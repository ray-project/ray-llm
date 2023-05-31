from aviary.api.sdk import models
from aviary.common.constants import MODELS


def test_model_descriptions_complete():
    all_backend_models = models()
    all_frontend_models = MODELS.keys()

    print(all_backend_models)

    assert len(all_backend_models) == len(all_frontend_models)
    assert sorted(all_backend_models) == sorted(all_frontend_models)
