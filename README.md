# RayLLM - LLMs on Ray

[![Build status](https://badge.buildkite.com/d6d7af987d1db222827099a953410c4e212b32e8199ca513be.svg?branch=master)](https://buildkite.com/anyscale/aviary-docker)

Try it now: [🦜🔍 Ray Aviary Explorer 🦜🔍](http://aviary.anyscale.com/)

RayLLM (formerly known as Aviary) is an LLM serving solution that makes it easy to deploy and manage
a variety of open source LLMs, built on [Ray Serve](https://docs.ray.io/en/latest/serve/index.html). It does this by: 

- Providing an extensive suite of pre-configured open source LLMs, with defaults that work out of the box.
- Supporting Transformer models hosted on [Hugging Face Hub](http://hf.co) or present on local disk.
- Simplifying the deployment of multiple LLMs
- Simplifying the addition of new LLMs
- Offering unique autoscaling support, including scale-to-zero.
- Fully supporting multi-GPU & multi-node model deployments.
- Offering high performance features like continuous batching, quantization and streaming.
- Providing a REST API that is similar to OpenAI's to make it easy to migrate and cross test them.

In addition to LLM serving, it also includes a CLI and a web frontend (Aviary Explorer) that you can use to compare the outputs of different models directly, rank them by quality, get a cost and latency estimate, and more. 

RayLLM supports continuous batching by integrating with [Hugging Face text-generation-inference (based off Apache 2.0-licensed fork)](https://github.com/Yard1/text-generation-inference) and [vLLM](https://github.com/vllm-project/vllm). Continuous batching allows you to get much better throughput and latency than static batching.

RayLLM leverages [Ray Serve](https://docs.ray.io/en/latest/serve/index.html), which has native support for autoscaling 
and multi-node deployments. RayLLM can scale to zero and create
new model replicas (each composed of multiple GPU workers) in response to demand.


# Getting started

## Deploying RayLLM 

The guide below walks you through the steps required for deployment of RayLLM on Ray Serve.

### Locally

We highly recommend using the official `anyscale/aviary` Docker image to run RayLLM. Manually installing RayLLM is currently not a supported use-case due to specific dependencies required, some of which are not available on pip.


```shell
cache_dir=${XDG_CACHE_HOME:-$HOME/.cache}

docker run -it --gpus all --shm-size 1g -p 8000:8000 -e HF_HOME=~/data -v $cache_dir:~/data anyscale/aviary:latest bash
# Inside docker container
aviary run --model ~/models/continuous_batching/amazon--LightGPT.yaml
```

### On a Ray Cluster

RayLLM uses Ray Serve, so it can be deployed on Ray Clusters.

Currently, we only have a guide and pre-configured YAML file for AWS deployments.
**Make sure you have exported your AWS credentials locally.**

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...
```

Start by cloning this repo to your local machine.

You may need to specify your AWS private key in the `deploy/ray/aviary-cluster.yaml` file.
See [Ray on Cloud VMs](https://docs.ray.io/en/latest/cluster/vms/index.html) page in
Ray documentation for more details.

```shell
git clone https://github.com/ray-project/ray-llm.git
cd ray-llm

# Start a Ray Cluster (This will take a few minutes to start-up)
ray up deploy/ray/aviary-cluster.yaml
```

#### Connect to your Cluster

```shell
# Connect to the Head node of your Ray Cluster (This will take several minutes to autoscale)
ray attach deploy/ray/aviary-cluster.yaml

# Deploy the LightGPT model. 
serve run serve/amazon--LightGPT.yaml
```

You can deploy any model in the `models` directory of this repo, 
or define your own model YAML file and run that instead.

### On Kubernetes

For Kubernetes deployments, please see our extensive documentation for [deploying Ray Serve on KubeRay](https://docs.ray.io/en/latest/serve/production-guide/kubernetes.html).
## Query your models

Once the models are deployed, you can install a client outside of the Docker container to query the backend.

```shell
pip install "aviary @ git+https://github.com/ray-project/ray-llm.git"
```

You can query your RayLLM deployment in many ways.

In all cases start out by doing: 

```shell
export ENDPOINT_URL="http://localhost:8000/v1"
```

This is because your deployment is running locally, but you can also access remote deployments (in which case you would set `ENDPOINT_URL` to a remote URL).

### Using curl

You can use curl at the command line to query your deployed LLM: 

```shell
% curl $ENDPOINT_URL/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}],
    "temperature": 0.7
  }'
```
```text
{
  "id":"meta-llama/Llama-2-7b-chat-hf-308fc81f-746e-4682-af70-05d35b2ee17d",
  "object":"text_completion","created":1694809775,
  "model":"meta-llama/Llama-2-7b-chat-hf",
  "choices":[
    {
      "message":
        {
          "role":"assistant",
          "content":"Hello there! *adjusts glasses* It's a pleasure to meet you! Is there anything I can help you with today? Have you got a question or a task you'd like me to assist you with? Just let me know!"
        },
      "index":0,
      "finish_reason":"stop"
    }
  ],
  "usage":{"prompt_tokens":30,"completion_tokens":53,"total_tokens":83}}
```

### Connecting directly over python

Use the `requests` library to connect with Python. Use this script to receive a streamed response, automatically parse the outputs, and print just the content.

```python
import os
import json
import requests

s = requests.Session()

api_base = os.getenv("ENDPOINT_URL")
url = f"{api_base}/chat/completions"
body = {
  "model": "meta-llama/Llama-2-7b-chat-hf",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me a long story with many words."}
  ],
  "temperature": 0.7,
  "stream": True,
}

with s.post(url, json=body, stream=True) as response:
    for chunk in response.iter_lines(decode_unicode=True):
        if chunk is not None:
            try:
                # Get data from reponse chunk
                chunk_data = chunk.split("data: ")[1]

                # Get message choices from data
                choices = json.loads(chunk_data)["choices"]

                # Pick content from first choice
                content = choices[0]["delta"]["content"]

                print(content, end="", flush=True)
            except json.decoder.JSONDecodeError:
                # Chunk was not formatted as expected
                pass
            except KeyError:
                # No message was contained in the chunk
                pass
    print("")
```

### Using the OpenAI SDK

RayLLM uses an OpenAI-compatible API, allowing us to use the OpenAI
SDK to access our deployments. To do so, we need to set the `OPENAI_API_BASE` env var. 


```shell
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY='not_a_real_key'
```

```python
import openai

# List all models.
models = openai.Model.list()
print(models)

# Note: not all arguments are currently supported and will be ignored by the backend.
chat_completion = openai.ChatCompletion.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Say 'test'."}
    ],
    temperature=0.7
)
print(chat_completion)
```


# RayLLM Reference

## Installing RayLLM

To install RayLLM and its dependencies, run the following command:

```shell
pip install "aviary @ git+https://github.com/ray-project/ray-llm.git"
```

RayLLM consists of a set of configurations and utilities for deploying LLMs on Ray Serve,
in addition to a frontend (Aviary Explorer), both of which come with additional
dependencies. To install the dependencies for the frontend run the following commands:

```shell
pip install "aviary[frontend] @ git+https://github.com/ray-project/ray-llm.git"
```

The backend dependencies are heavy weight, and quite large. We recommend using the official
`anyscale/aviary` image. Installing the backend manually is not a supported usecase.

## Running Aviary Explorer locally

The frontend is a [Gradio](https://gradio.app/) interface that allows you to interact
with the models in the backend through a web interface.
The Gradio app is served using [Ray Serve](https://docs.ray.io/en/latest/serve/index.html).

To run the Aviary Explorer locally, you need to set the following environment variable:

```shell
export ENDPOINT_URL=<hostname of the backend, eg. 'http://localhost:8000'>
```

Once you have set these environment variables, you can run the frontend with the
following command:

```shell
serve run aviary.frontend.app:app --non-blocking
```

You will be able to access it at `http://localhost:8000/frontend` in your browser.

To just use the Gradio frontend without Ray Serve, you can start it 
with `python aviary/frontend/app.py`. In that case, the Gradio interface should be accessible at `http://localhost:7860` in your browser.
If running the frontend yourself is not an option, you can still use 
[our hosted version](http://aviary.anyscale.com/) for your experiments.

Note that the frontend will not dynamically update the list of models should they change in the backend. In order for the frontend to update, you will need to restart it.

### Usage stats collection

Ray collects basic, non-identifiable usage statistics to help us improve the project.
For more information on what is collected and how to opt-out, see the
[Usage Stats Collection](https://docs.ray.io/en/latest/cluster/usage-stats.html) page in
Ray documentation.

## Using RayLLM through the CLI

RayLLM uses the Ray Serve CLI that allows you to interact with deployed models.


```shell
# Start a new model in Ray Serve from provided configuration
serve run serve/<model_config_path>

# Get the status of the running deployments
serve status

# Get the current config of current live Serve applications 
serve config

# Shutdown all Serve applications
serve shutdown
```


## RayLLM Model Registry

You can easily add new models by adding two configuration files.
To learn more about how to customize or add new models, 
see the [Model Registry](models/README.md).

# Frequently Asked Questions

## How do I add a new model?

The easiest way is to copy the configuration of the existing model's YAML file and modify it. See models/README.md for more details.

## How do I deploy multiple models at once?

Run multiple models at once by aggregating the Serve configs for different models into a single, unified config. For example, use this config to run the `LightGPT` and `Llama-2-7b-chat` model in a single Serve application:

```yaml
# File name: serve/config.yaml

applications:
- name: router
  route_prefix: /
  import_path: aviary.backend:router_application
  args:
    models:
      amazon/LightGPT: ./models/continuous_batching/amazon--LightGPT.yaml
      meta-llama/Llama-2-7b-chat-hf: ./models/continuous_batching/meta-llama--Llama-2-7b-chat-hf.yaml
- name: amazon--LightGPT
  route_prefix: /amazon--LightGPT
  import_path: aviary.backend:llm_application
  args:
    model: "./models/continuous_batching/amazon--LightGPT.yaml"
- name: meta-llama--Llama-2-7b-chat-hf
  route_prefix: /meta-llama--Llama-2-7b-chat-hf
  import_path: aviary.backend:llm_application
  args:
    model: "./models/continuous_batching/meta-llama--Llama-2-7b-chat-hf.yaml"
```

The config includes both models in the `model` argument for the `router`. Additionally, the Serve configs for both model applications are included. Save this unified config file to the `serve/` folder.

Run the config to deploy the models:

```shell
serve run serve/<config.yaml>
```

## How do I deploy a model to multiple nodes?

All our default model configurations enforce a model to be deployed on one node for high performance. However, you can easily change this if you want to deploy a model across nodes for lower cost or GPU availability. In order to do that, go to the YAML file in the model registry and change `placement_strategy` to `PACK` instead of `STRICT_PACK`.

## My deployment isn't starting/working correctly, how can I debug?

There can be several reasons for the deployment not starting or not working correctly. Here are some things to check:
1. You might have specified an invalid model id.
2. Your model may require resources that are not available on the cluster. A common issue is that the model requires Ray custom resources (eg. `accelerator_type_a10`) in order to be scheduled on the right node type, while your cluster is missing those custom resources. You can either modify the model configuration to remove those custom resources or better yet, add them to the node configuration of your Ray cluster. You can debug this issue by looking at Ray Autoscaler logs ([monitor.log](https://docs.ray.io/en/latest/ray-observability/user-guides/configure-logging.html#system-component-logs)).
3. Your model is a gated Hugging Face model (eg. meta-llama). In that case, you need to set the `HUGGING_FACE_HUB_TOKEN` environment variable cluster-wide. You can do that either in the Ray cluster configuration or by setting it before running `serve run`
4. Your model may be running out of memory. You can usually spot this issue by looking for keywords related to "CUDA", "memory" and "NCCL" in the replica logs or `serve run` output. In that case, consider reducing the `max_batch_prefill_tokens` and `max_batch_total_tokens` (if applicable). See models/README.md for more information on those parameters.

In general, [Ray Dashboard](https://docs.ray.io/en/latest/serve/monitoring.html#ray-dashboard) is a useful debugging tool, letting you monitor your Ray Serve / LLM application and access Ray logs.

A good sanity check is deploying the test model in tests/models/. If that works, you know you can deploy _a_ model. 

### How do I write a program that accesses both OpenAI and your hosted model at the same time? 

The OpenAI `create()` commands allow you to specify the `API_KEY` and `API_BASE`. So you can do something like this. 

```python
# Call your self-hosted model running on the local host:
OpenAI.ChatCompletion.create(api_base="http://localhost:8000/v1", api_key="",...)

# Call OpenAI. Set OPENAI_API_KEY to your key and unset OPENAI_API_BASE 
OpenAI.ChatCompletion.create(api_key="OPENAI_API_KEY", ...)
```

## Getting Help and Filing Bugs / Feature Requests

We are eager to help you get started with RayLLM. You can get help on: 

- Via Slack -- fill in [this form](https://docs.google.com/forms/d/e/1FAIpQLSfAcoiLCHOguOm8e7Jnn-JJdZaCxPGjgVCvFijHB5PLaQLeig/viewform) to sign up. 
- Via [Discuss](https://discuss.ray.io/c/llms-generative-ai/27). 

For bugs or for feature requests, please submit them [here](https://github.com/ray-project/ray-llm/issues/new).

## Contributions

We are also interested in accepting contributions. Those could be anything from a new evaluator, to integrating a new model with a yaml file, to more.
Feel free to post an issue first to get our feedback on a proposal first, or just file a PR and we commit to giving you prompt feedback.

We use `pre-commit` hooks to ensure that all code is formatted correctly.
Make sure to `pip install pre-commit` and then run `pre-commit install`.
You can also run `./format` to run the hooks manually.