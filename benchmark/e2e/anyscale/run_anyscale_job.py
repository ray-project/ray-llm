import hashlib
import os
import signal
import subprocess
import sys
import time
from functools import wraps
from typing import List, Optional

import typer

app = typer.Typer()


def _exponential_backoff_retry(
    initial_retry_delay_s: float,
    max_retries: int,
):
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_cnt = 0
            retry_delay_s = initial_retry_delay_s
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retry_cnt += 1
                    if retry_cnt > max_retries:
                        raise
                    print(
                        f"Retry function call failed due to {e} "
                        f"in {retry_delay_s} seconds..."
                    )
                    time.sleep(retry_delay_s)
                    retry_delay_s *= 2

        return wrapper

    return inner


def _run(command: str, _return_stderr=False, check=True, cwd=None, *args, **kwargs):
    """Run a command in the shell"""
    print("+", command)
    env = kwargs.get("env", os.environ.copy())
    env["PYTHONUNBUFFERED"] = "1"
    res = subprocess.run(
        command,
        *args,
        shell=True,
        check=check,
        cwd=cwd,
        env=env,
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


def _file_hash(path: str):
    """Get the hash of a file"""
    with open(path, "r") as f:
        return hashlib.md5(f.read().encode("utf-8")).hexdigest()


def _create_cluster_compute(cc_path: str):
    cc_file_hash = _file_hash(cc_path)
    cc_name = f"aviary-{cc_file_hash}"

    _run(
        f"anyscale compute-config get {cc_name} || anyscale compute-config create -n {cc_name} {cc_path}"
    )
    return cc_name


def _create_cluster_env(ce_path: str, docker_image: str, smoke_test: bool = False):
    with open(ce_path, "r") as f:
        ce_file = f.read()
    ce_file = ce_file.replace("::param:docker_image::", docker_image).replace(
        "::param:smoke_test::", str(int(smoke_test))
    )
    new_ce_file = f"{ce_path}.tmp"
    with open(new_ce_file, "w") as f:
        f.write(ce_file)
    ce_name = f"aviary-{_file_hash(new_ce_file)}"
    _run(
        f"anyscale cluster-env get {ce_name} || anyscale cluster-env build -n {ce_name} {new_ce_file}"
    )
    return ce_name


def _create_job(
    job_path: str,
    cc_name: str,
    ce_name: str,
    working_dir: str,
    benchmark_scripts: List[str],
    name: str,
):
    with open(job_path, "r") as f:
        job_file = f.read()
    benchmark_scripts = "; ".join(benchmark_scripts)
    job_file = (
        job_file.replace("::param:compute_config::", cc_name)
        .replace("::param:cluster_env::", ce_name)
        .replace("::param:working_dir::", os.path.abspath(working_dir))
        .replace("::param:name::", name)
        .replace("::param:benchmark_scripts::", benchmark_scripts)
    )
    new_job_path = f"{job_path}_{name}.tmp"
    with open(new_job_path, "w") as f:
        f.write(job_file)
    return new_job_path


@_exponential_backoff_retry(initial_retry_delay_s=10, max_retries=3)
def _get_logs(job_id: str) -> str:
    ret = _run(f"anyscale job logs --id '{job_id}'", capture_output=True)
    assert ret
    return ret


def _get_job_id(job_name: str) -> str:
    ret = _run(
        f"anyscale job list --include-all-users --name '{job_name}'",
        capture_output=True,
    )
    return ret.splitlines()[3].split()[1]


job_name = ""
job_id = ""


def signal_handler(signum, frame):
    signal.signal(signum, signal.SIG_IGN)
    if job_id:
        _run(f"anyscale job terminate --id '{job_id}'")
        time.sleep(1)
    sys.exit(0)


@app.command(name="run")
def run(
    compute_config: str,
    cluster_env_template: str,
    job_template: str,
    working_dir: str,
    docker_image: str,
    name: str,
    benchmark_scripts: List[str],
    results_path: Optional[str],
):
    global job_name, job_id
    job_name = name
    cluster_compute = _create_cluster_compute(compute_config)
    cluster_env = _create_cluster_env(
        cluster_env_template,
        docker_image,
        os.environ.get("AVIARY_SMOKE_TEST", "0") != "0",
    )
    signal.signal(signal.SIGINT, signal_handler)
    job_file = _create_job(
        job_template, cluster_compute, cluster_env, working_dir, benchmark_scripts, name
    )
    try:
        _run(f"anyscale job submit '{job_file}'")
        time.sleep(1)
        job_id = _get_job_id(job_name)
        _run(f"anyscale job logs --id '{job_id}' --follow")
        _run(f"anyscale job wait --id '{job_id}'")
        exc = None
    except Exception as e:
        exc = e
    if results_path:
        with open(results_path, "w") as f:
            f.write(_get_logs(job_id))
    if exc:
        raise exc


if __name__ == "__main__":
    app()


"""
python /home/ray/default/aviary/benchmark/e2e/anyscale/run_anyscale_job.py \
    /home/ray/default/aviary/benchmark/e2e/anyscale/compute-config-g5-4xlarge.yaml \
    /home/ray/default/aviary/benchmark/e2e/anyscale/cluster-env-template.yaml \
    /home/ray/default/aviary/benchmark/e2e/anyscale/anyscale-job-template.yaml \
    /home/ray/default/aviary \
    anyscale/aviary:latest-tgi aviary-benchmark-$BUILDKITE_JOB_ID \
    './benchmark_uniform falcon-7b-tp1.yaml tiiuae/falcon-7b 1024'
"""
