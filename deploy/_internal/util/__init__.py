import enum
import hashlib
import subprocess
from typing import List, Optional

import yaml
from anyscale import AnyscaleSDK
from anyscale.controllers.service_controller import ServiceController
from pydantic import BaseModel, validator

service_controller = ServiceController()
sdk: AnyscaleSDK = service_controller.anyscale_api_client


class DeployPhase(str, enum.Enum):
    build = "build"
    backend = "backend"
    frontend = "frontend"
    all = "all"


class Config(BaseModel):
    deploy_env: str = "staging"
    deploy_phases: List[DeployPhase] = [DeployPhase.all]

    @property
    def ensure_git_clean(self):
        return self._is_prod_or_staging()

    def _is_prod_or_staging(self):
        return self.deploy_env in ["prod", "staging"]

    def get_project_name(self):
        return f"aviary-{self.deploy_env}" if self._is_prod_or_staging() else "aviary"

    def get_project_id(self):
        if self.deploy_env == "staging":
            return "prj_s7ky22b64qrx2yqumji46plg6x"
        elif self.deploy_env == "prod":
            return "prj_yv8vw373zcak4nrpjjmpn2y44f"
        else:
            return None

    def _get_service_version(self):
        if self._is_prod_or_staging():
            _assert_git_clean()
            return _git_version()
        return None

    @property
    def frontend_service_name(self):
        return f"aviary-oss-frontend-primary-{self.deploy_env}"

    @classmethod
    @validator("deploy_env", pre=True, always=True)
    def validate_git_clean(cls, deploy_env):
        if deploy_env in ["prod", "staging"]:
            _assert_git_clean()
        return deploy_env


# The entrypoint is responsible for populating this
global_config: Optional[Config] = None


def get_global_config():
    global global_config
    assert global_config, "Global config was never set"
    return global_config


def set_global_config(config: Config):
    global global_config
    assert not global_config, "Global config was already set"
    global_config = config
    return config


# Utils
def _read_yaml(file: str):
    with open(file, "r") as stream:
        return yaml.safe_load(stream)


def _file_hash(path: str):
    """Get the hash of a file"""
    with open(path, "r") as f:
        return hashlib.md5(f.read().encode("utf-8")).hexdigest()


# Git utils
def _git_version():
    """Get the git version of the repo"""
    return _run("git rev-parse HEAD", capture_output=True)


def _assert_git_clean():
    try:
        return _run("git diff-index --quiet HEAD --")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "Please make sure you are on the latest master. You have unresolved git changes."
        ) from e


def _run(command: str, _return_stderr=False, check=True, cwd=None, *args, **kwargs):
    """Run a command in the shell"""
    print("+", command)
    res = subprocess.run(
        command,
        *args,
        shell=True,
        check=check,
        cwd=cwd,
        **kwargs,
    )
    if kwargs.get("capture_output"):
        if _return_stderr:
            return (
                res.stdout.decode("utf-8").strip(),
                res.stderr.decode("utf-8").strip(),
            )
        return res.stdout.decode("utf-8").strip()

    return res


def _get_service_hostname_and_token(service_name: str):
    service_id = service_controller.get_service_id(service_name=service_name)
    service = sdk.get_service_v2(service_id=service_id).result
    hostname = service.hostname
    token = service.auth_token
    return hostname, token


def _rollout(filename: str, canary_percent: Optional[int] = None, ask_confirm=True):
    c = ""
    if canary_percent is not None:
        c = f"--canary-percent {canary_percent}"
    _run(f"""anyscale service rollout -f "{filename}" {c}""")
    if ask_confirm:
        c = input(
            "Please confirm that the new version is healthy, then press y to continue (y/n): "
        )
        assert c == "y", "Aborting deploy. Please rollback manually."


def _create_cluster_compute(cc_path: str):
    cc_file_hash = _file_hash(cc_path)
    cc_name = f"aviary-{cc_file_hash}"

    _run(
        f"anyscale compute-config get {cc_name} || anyscale compute-config create -n {cc_name} {cc_path}"
    )
    return cc_name


def _create_cluster_env(ce_path: str):
    ce_file_hash = _file_hash(ce_path)
    ce_name = f"aviary-{ce_file_hash}"
    _run(
        f"anyscale cluster-env get {ce_name} || anyscale cluster-env build -n {ce_name} {ce_path}"
    )
    return ce_name
