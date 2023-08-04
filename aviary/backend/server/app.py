import os

from ray import serve

from aviary.backend.logger import get_logger
from aviary.backend.server.model_app import LLMDeployment, model_app
from aviary.backend.server.router_app import Router, router_app

logger = get_logger(__name__)

TextGenerationInferenceLLMDeployment = serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 2,
        "max_replicas": 8,
    },
    max_concurrent_queries=10,  # Maximum backlog for a single replica
    health_check_period_s=10,
    health_check_timeout_s=30,
)(serve.ingress(model_app)(LLMDeployment))


RouterDeployment = serve.deployment(
    route_prefix="/",
    # TODO make this configurable in aviary run
    autoscaling_config={
        "min_replicas": int(os.environ.get("AVIARY_ROUTER_MIN_REPLICAS", 2)),
        "initial_replicas": int(os.environ.get("AVIARY_ROUTER_INITIAL_REPLICAS", 2)),
        "max_replicas": int(os.environ.get("AVIARY_ROUTER_MAX_REPLICAS", 16)),
        "target_num_ongoing_requests_per_replica": int(
            os.environ.get("AVIARY_ROUTER_TARGET_NUM_ONGOING_REQUESTS_PER_REPLICA", 200)
        ),
    },
    max_concurrent_queries=1000,  # Maximum backlog for a single replica
)(serve.ingress(router_app)(Router))
