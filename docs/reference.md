
# Aviary Reference

## Installing Aviary

To install Aviary and its dependencies, run the following command:

```shell
pip install "aviary @ git+https://github.com/ray-project/aviary.git"
```

The default Aviary installation only includes the Aviary API client.

Aviary consists of a backend and a frontend (Aviary Explorer), both of which come with additional
dependencies. To install the dependencies for the frontend run the following commands:

```shell
pip install "aviary[frontend] @ git+https://github.com/ray-project/aviary.git"
```

The backend dependencies are heavy weight, and quite large. We recommend using the official
`anyscale/aviary` image. Installing the backend manually is not a supported usecase.

## Running Aviary Frontend locally

Aviary consists of two components, a backend and a frontend.
The Backend exposes a Ray Serve FastAPI interface running on a Ray cluster allowing you to deploy various LLMs efficiently.

The frontend is a [Gradio](https://gradio.app/) interface that allows you to interact
with the models in the backend through a web interface.
The Gradio app is served using [Ray Serve](https://docs.ray.io/en/latest/serve/index.html).

To run the Aviary frontend locally, you need to set the following environment variable:

```shell
export AVIARY_URL=<hostname of the backend, eg. 'http://localhost:8000'>
```

Once you have set these environment variables, you can run the frontend with the
following command:

```shell
serve run aviary.frontend.app:app
```

To just use the Gradio frontend without Ray Serve, you can start it 
with `python aviary/frontend/app.py`.

In any case, the Gradio interface should be accessible at `http://localhost:7860`
in your browser.
If running the frontend yourself is not an option, you can still use 
[our hosted version](http://aviary.anyscale.com/) for your experiments.

### Usage stats collection

Aviary backend collects basic, non-identifiable usage statistics to help us improve the project.
The mechanism for collection is the same as in Ray.
For more information on what is collected and how to opt-out, see the
[Usage Stats Collection](https://docs.ray.io/en/latest/cluster/usage-stats.html) page in
Ray documentation.

## Aviary Model Registry

Aviary allows you to easily add new models by adding a single configuration file.
To learn more about how to customize or add new models, 
see the [Aviary Model Registry](models/README.md).
