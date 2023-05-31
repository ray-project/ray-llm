import os

from util import (
    Config,
    _create_cluster_compute,
    _create_cluster_env,
    _rollout,
)


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
        self.ce_path = os.path.join(self.deploy_dir, "cluster-env.yaml")
        self.service_start_path = os.path.join(self.deploy_dir, "service.yaml")
        self.service_final_path = os.path.join(self.deploy_dir, "service.yaml.tmp")

    def build(self):
        cc_name = _create_cluster_compute(self.cc_path)
        ce_name = _create_cluster_env(self.ce_path)
        version = self.config._get_service_version()
        project_id = self.config.get_project_id()
        with open(self.service_start_path, "r") as f:
            config = (
                f.read()
                + f"\ncompute_config: {cc_name}"
                + f"\nproject_id: {project_id}"
                + f"\ncluster_env: {ce_name}:1"
                + f"\nname: {self.backend_service_name}"
                + (f"\nversion: {version}" if version else "")
                + "\n"
            )

        with open(self.service_final_path, "w") as f:
            f.write(config)

    def deploy(self):
        ask_confirm = self.config._is_prod_or_staging()
        if ask_confirm:
            _rollout(self.service_final_path, 0, ask_confirm=ask_confirm)
            _rollout(self.service_final_path, 100, ask_confirm=ask_confirm)
        _rollout(self.service_final_path, canary_percent=None, ask_confirm=ask_confirm)
