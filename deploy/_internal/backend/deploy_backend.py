import os
from copy import deepcopy

import yaml
from util import (
    Config,
    _create_cluster_compute,
    _create_cluster_env,
    _rollout,
)

from aviary.backend.server.utils import parse_args


class BackendController:
    def __init__(self, config: Config):
        self.config = config
        self.backend_service_name = (
            f"aviary-oss-backend-primary-{self.config.deploy_env}"
        )

        # Files
        self.deploy_dir = os.path.dirname(os.path.realpath(__file__))

        self.cc_filename = (
            "compute-config-prod.yaml"
            if self.config._is_prod_or_staging()
            else "compute-config.yaml"
        )
        self.cc_path = os.path.join(self.deploy_dir, self.cc_filename)
        # TODO make this configurable
        self.ce_path = os.path.join(self.deploy_dir, "cluster-env-tgi.yaml")
        self.service_start_path = os.path.join(self.deploy_dir, "service.yaml")
        self.service_final_path = os.path.join(self.deploy_dir, "service.yaml.tmp")

    def build(self):
        cc_name = _create_cluster_compute(self.cc_path)
        ce_name = _create_cluster_env(self.ce_path)
        version = self.config._get_service_version()
        project_id = self.config.get_project_id()
        with open(self.service_start_path, "r") as f:
            service = yaml.load(f, Loader=yaml.SafeLoader)

        service["compute_config"] = cc_name
        service["project_id"] = project_id
        service["cluster_env"] = f"{ce_name}:1"
        service["name"] = self.backend_service_name
        service["version"] = version if version else ""
        models = parse_args(service.pop("models"))
        applications = service["ray_serve_config"]["applications"]
        llm_application_template = next(
            app for app in (applications) if app["name"] == "::param:app_name::"
        )
        router_application_template = next(
            app for app in (applications) if app["name"] == "router"
        )

        def build_llm_application(model, template):
            app = deepcopy(template)
            app["name"] = model.model_config.model_id.replace("/", "--").replace(
                ".", "_"
            )
            app["route_prefix"] = "/" + app["name"]
            app["args"] = {"model": model.yaml()}
            return app

        def build_router(configs, template):
            app = deepcopy(template)
            app["args"] = {
                "model_configs": {
                    model.model_config.model_id: model.yaml() for model in configs
                }
            }
            return app

        applications = [
            build_llm_application(model, llm_application_template) for model in models
        ]
        applications += [build_router(models, router_application_template)]

        service["ray_serve_config"]["applications"] = applications

        with open(self.service_final_path, "w") as f:
            yaml.dump(service, f)

    def deploy(self):
        ask_confirm = self.config._is_prod_or_staging()
        if ask_confirm:
            _rollout(self.service_final_path, 0, ask_confirm=ask_confirm)
            _rollout(self.service_final_path, 100, ask_confirm=ask_confirm)
        _rollout(self.service_final_path, canary_percent=None, ask_confirm=ask_confirm)
