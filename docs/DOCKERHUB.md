<!---
Docker Hub Description File
-->

# Overview

This is the publicly available set of Docker images for Anyscale/Ray's RayLLM (formerly Aviary) project.

RayLLM is an LLM serving solution that makes it easy to deploy and manage a variety of open source LLMs. It does this by:

- Providing an extensive suite of pre-configured open source LLMs, with defaults that work out of the box.
- Supporting Transformer models hosted on Hugging Face Hub or present on local disk.
- Simplifying the deployment of multiple LLMs within a single unified framework.
- Simplifying the addition of new LLMs to within minutes in most cases.
- Offering unique autoscaling support, including scale-to-zero.
- Fully supporting multi-GPU & multi-node model deployments.
- Offering high performance features like continuous batching, quantization and streaming.
- Providing a REST API that is similar to OpenAI's to make it easy to migrate and cross test them.

[Read more here](https://github.com/ray-project/ray-llm)

## Tags

| Name | Notes |
|----|----|
| [`:0.3.1`](https://hub.docker.com/layers/anyscale/ray-llm/0.3.1/images/sha256-0dad10786076e18530fbd8016929ab9b240c8fe12163d5e74d8784ff1cbf5fb4) | Release v0.3.1 |
| [`:0.3.0`](https://hub.docker.com/layers/anyscale/ray-llm/0.3.0/images/sha256-310df8d6bfcce49fa00c0040f090099b7d376ed9535df85fa4147e7c159e7e90) | Release v0.3.0 |
| `:latest` | Most recently pushed version release image |

## Usage

See: [ray-project/ray-llm "Deploying RayLLM"](https://github.com/ray-project/ray-llm#deploying-rayllm) for full instructions

### Example

Requires a machine with compatible NVIDIA A10 GPU and valid `HUGGING_FACE_HUB_TOKEN` to run the [Amazon LightGPT model](https://huggingface.co/amazon/LightGPT):

```sh
docker run \
    --gpus all \
    -e HUGGING_FACE_HUB_TOKEN=<your_token> \
    --shm-size 1g \
    -p 8000:8000 \
    --entrypoint rayllm \
    anyscale/rayllm:latest run --model models/continuous_batching/amazon--LightGPT.yaml
```

# Source

Source is available at https://github.com/ray-project/ray-llm

