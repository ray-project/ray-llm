import typer
from backend.deploy_backend import BackendController
from frontend.deploy_frontend import FrontendController
from util import Config, _get_service_hostname_and_token

app = typer.Typer()


@app.command(name="execute")
def execute(env: str, service: str):
    deploy_frontend = service == "frontend"
    deploy_backend = service == "backend"

    config = Config(deploy_env=env)
    if deploy_backend:
        controller = BackendController(config)
        controller.deploy()
    elif deploy_frontend:
        controller = FrontendController(config)
        controller.deploy()


@app.command(name="build")
def build(env: str, service: str):
    build_frontend = service == "frontend"
    build_backend = service == "backend"
    config = Config(deploy_env=env)

    b_controller = BackendController(config)
    f_controller = FrontendController(config)

    if build_backend:
        b_controller.build()

    elif build_frontend:
        try:
            backend_hostname, backend_token = _get_service_hostname_and_token(
                b_controller.backend_service_name
            )
        except Exception as e:
            raise RuntimeError(
                "Could not find the backend service for this environment. Please deploy the backend first."
            ) from e

        f_controller.build(backend_hostname, backend_token)

    print()
    print("****")
    print(f"Build completed successfully. Env: {env}, service: {service}")
    print(f"To deploy {env}: ./deploy.sh execute {env} {service}")
    print("****")


if __name__ == "__main__":
    app()
