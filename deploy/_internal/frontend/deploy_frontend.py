import os
from typing import Optional

from util import (
    Config,
    _create_cluster_compute,
    _create_cluster_env,
    _get_service_hostname_and_token,
    _rollout,
    _run,
)


class FrontendController:
    def __init__(
        self,
        config: Config,
    ):
        self.config = config
        self.frontend_service_name = (
            f"aviary-oss-frontend-primary-{self.config.deploy_env}"
        )

        # Files
        self.deploy_dir = os.path.dirname(os.path.realpath(__file__))

        self.service_config_filename = f"{self.deploy_dir}/service.yaml.tmp"

        cc_filename = (
            "compute-config-prod.yaml"
            if self.config._is_prod_or_staging()
            else "compute-config.yaml"
        )
        self.cc_path = os.path.join(self.deploy_dir, cc_filename)
        self.ce_path = os.path.join(self.deploy_dir, "cluster-env.yaml")
        self.cf_path = os.path.join(self.deploy_dir, "cloudfront.json")

    def deploy(self):
        self._rollout_frontend_service()
        frontend_hostname, frontend_token = _get_service_hostname_and_token(
            self.frontend_service_name
        )
        self._create_cloudfront(frontend_hostname, frontend_token)

    def build(self, backend_hostname: str, backend_token: str, protocol: str = "https"):
        ce_name = _create_cluster_env(self.ce_path)
        cc_name = _create_cluster_compute(self.cc_path)

        version = self.config._get_service_version()
        project_id = self.config.get_project_id()
        with open(f"{self.deploy_dir}/service.yaml", "r") as f:
            config = (
                f.read()
                .replace("::param:aviary_url::", f"{protocol}://{backend_hostname}")
                .replace("::param:aviary_token::", backend_token)
                + f"\ncompute_config: {cc_name}"
                + f"\nproject_id: {project_id}"
                + f"\ncluster_env: {ce_name}:1"
                + (f"\nversion: {version}" if version else "")
                + f"\nname: {self.frontend_service_name}"
                + "\n"
            )
        with open(f"{self.deploy_dir}/service.yaml.tmp", "w") as f:
            f.write(config)

    def _rollout_frontend_service(self):
        ask_confirm = self.config._is_prod_or_staging()
        if ask_confirm:
            self._rollout(canary_percent=0, ask_confirm=ask_confirm)
            self._rollout(canary_percent=100, ask_confirm=ask_confirm)

        self._rollout(ask_confirm=ask_confirm)

    def _rollout(self, canary_percent: Optional[int] = None, ask_confirm: bool = True):
        return _rollout(
            self.service_config_filename,
            ask_confirm=ask_confirm,
            canary_percent=canary_percent,
        )

    def _create_cloudfront(self, hostname: str, token: str):
        with open(self.cf_path, "r") as f:
            data = (
                f.read()
                .replace("::param:hostname::", hostname)
                .replace("::param:token::", token)
            )

        with open(self.deploy_dir + "/cloudfront.json.tmp", "w") as f:
            f.write(data)

        stdout, stderr = _run(
            f"aws cloudfront create-distribution --cli-input-json file://{self.deploy_dir}/cloudfront.json.tmp | grep DomainName",
            _return_stderr=True,
            capture_output=True,
            check=False,
        )
        if "Already exists: " in stderr:
            cf_id = stderr.split("Already exists: ")[1].split(" ")[0].strip()
            print("Cloudfront already exists, skipping creation", cf_id)
            _run(
                f"aws cloudfront get-distribution --id {cf_id} | grep DomainName",
            )
        else:
            print(stdout)
