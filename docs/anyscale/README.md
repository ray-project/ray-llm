# Aviary - Study stochastic parrots in the wild

Aviary helps you deploy large language models (LLMs) with state-of-the-art optimizations on top of Ray Serve. 
In particular, it offers good support for Transformer models hosted on [Hugging Face](http://hf.co) and in many cases also 
supports [DeepSpeed](https://www.deepspeed.ai/) inference acceleration as well as continuous batching and paged attention. 
This template uses the `anyscale/aviary:latest-tgi` docker image.

In this guide, we will go over deploying a model locally using `serve run` as well as on an Anyscale Service. Alternatively, you can use the [Aviary CLI and OpenAI SDK](https://github.com/ray-project/aviary/tree/master#using-the-aviary-cli) on this workspace. The CLI can help you compare the outputs of different models directly, rank them by quality, get a cost and latency estimate, and more. 

# Deploy a Large Language Model 
From the terminal use the Ray Serve CLI to deploy a model:

```shell
# Deploy the LightGPT model. 

serve run docs/anyscale/serve.yaml
```

The serve YAML file runs the lightgpt model. You can modify it to deploy any model in the `models` directory of this repo, provided you have the right GPU resources. You can also define your own model YAML file in the `models/` directory and run that instead. Follow the Aviary Model Registry [guide](models/README.md) for that.

### Query the model

Run the following command in a separate terminal. 

```shell
python docs/anyscale/openai-sdk-query.py
```
```text
Output:
The top rated restaurants in San Francisco include:
 • Chez Panisse
 • Momofuku Noodle Bar
 • Nopa
 • Saison
 • Mission Chinese Food
 • Sushi Nakazawa
 • The French Laundry
 • Delfina
 • Spices
 • Quince
 • Bistro L'Etoile
 • The Slanted Door
 • The Counter
 • The Chronicle
 • The Mint
 • The French Press
 • The Palace Cafe
 • The Inn at the Opera House
 • The Green Table
 • The Palace Cafe
```

# Deploying on Anyscale Service

To deploy an application with one model on an Anyscale Service you can run:

```shell
anyscale service rollout -f docs/anyscale/service.yaml --name {ENTER_NAME_FOR_SERVICE_HERE}
```

This is setup to run the amazon/LightGPT model, but can be easily modified to run any of the other models in this repo.
In order to query the endpoint, you can modify the `docs/anyscale/openai-sdk-query.py` script, replacing the query url and tokens with the Service URL and tokens found in the Service UI.


# Aviary Model Registry

Aviary allows you to easily add new models by adding a single configuration file.
To learn more about how to customize or add new models, 
see the [Aviary Model Registry](models/README.md).

# Frequently Asked Questions

## How do I add a new model?

The easiest way is to copy the configuration of the existing model's YAML file and modify it. See models/README.md for more details.

## How do I deploy multiple models at once?

You can append another application configuration to the YAML in `docs/anyscale/serve.yaml` file. Alternatively, you can use the Aviary CLI linked above.

## How do I deploy a model to multiple nodes?

All our default model configurations enforce a model to be deployed on one node for high performance. However, you can easily change this if you want to deploy a model across nodes for lower cost or GPU availability. In order to do that, go to the YAML file in the model registry and change `placement_strategy` to `PACK` instead of `STRICT_PACK`.

## My deployment isn't starting/working correctly, how can I debug?

There can be several reasons for the deployment not starting or not working correctly. Here are some things to check:
1. You might have specified an invalid model id.
2. Your model may require resources that are not available on the cluster. A common issue is that the model requires Ray custom resources (eg. `accelerator_type_a10`) in order to be scheduled on the right node type, while your cluster is missing those custom resources. You can either modify the model configuration to remove those custom resources or better yet, add them to the node configuration of your Ray cluster. You can debug this issue by looking at Ray Autoscaler logs ([monitor.log](https://docs.ray.io/en/latest/ray-observability/user-guides/configure-logging.html#system-component-logs)).
3. Your model is a gated Hugging Face model (eg. meta-llama). In that case, you need to set the `HUGGING_FACE_HUB_TOKEN` environment variable cluster-wide. You can do that either in the Ray cluster configuration or by setting it before running `serve run`.
4. Your model may be running out of memory. You can usually spot this issue by looking for keywords related to "CUDA", "memory" and "NCCL" in the replica logs or `serve run` output. In that case, consider reducing the `max_batch_prefill_tokens` and `max_batch_total_tokens` (if applicable). See models/README.md for more information on those parameters.

In general, [Ray Dashboard](https://docs.ray.io/en/latest/serve/monitoring.html#ray-dashboard) is a useful debugging tool, letting you monitor your Aviary application and access Ray logs.

A good sanity check is deploying the test model in tests/models/. If that works, you know you can deploy _a_ model. 

# Getting Help and Filing Bugs / Feature Requests

We are eager to help you get started with Aviary. You can get help on: 

- Via Slack -- fill in [this form](https://docs.google.com/forms/d/e/1FAIpQLSfAcoiLCHOguOm8e7Jnn-JJdZaCxPGjgVCvFijHB5PLaQLeig/viewform) to sign up. 
- Via [Discuss](https://discuss.ray.io/c/llms-generative-ai/27). 

For bugs or for feature requests, please submit them [here](https://github.com/ray-project/aviary/issues/new).

We have people in both US and European time zones who will help answer your questions. 

