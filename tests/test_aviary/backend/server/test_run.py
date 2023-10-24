import pytest

from rayllm.backend.server.models import ScalingConfig
from rayllm.backend.server.run import set_deployment_placement_options


def test_set_deployment_placement_options():
    deployment_config = {
        "ray_actor_options": {"num_cpus": 2, "resources": {"custom_resource": 1}}
    }
    scaling_config = ScalingConfig(
        num_workers=2,
        resources_per_worker={"custom_resource_2": 1},
        placement_group_strategy="PACK",
    )
    deployment_config = set_deployment_placement_options(
        deployment_config, scaling_config
    )
    assert deployment_config["placement_group_bundles"] == [
        {"CPU": 2, "GPU": 0, "custom_resource": 1},
        {"GPU": 1, "CPU": 1, "custom_resource_2": 1},
        {"GPU": 1, "CPU": 1, "custom_resource_2": 1},
    ]
    assert deployment_config["placement_group_strategy"] == "PACK"

    deployment_config = {
        "ray_actor_options": {},
        "placement_group_bundles": [{"CPU": 2, "GPU": 0, "custom_resource": 1}],
    }
    with pytest.raises(ValueError):
        deployment_config = set_deployment_placement_options(
            deployment_config, scaling_config
        )

    deployment_config = {"ray_actor_options": {}, "placement_group_strategy": "PACK"}
    with pytest.raises(ValueError):
        deployment_config = set_deployment_placement_options(
            deployment_config, scaling_config
        )
