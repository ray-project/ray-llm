# RayLLM model registry

Each model is defined by a YAML configuration file in this directory.

## Modify an existing model

To modify an existing model, simply edit the YAML file for that model.
Each config file consists of three sections:

- `deployment_config`,
- `engine_config`,
- `scaling_config`.

It's best to check out examples of existing models to see how they are configured.

## Deployment config

The `deployment_config` section corresponds to
[Ray Serve configuration](https://docs.ray.io/en/latest/serve/production-guide/config.html)
and specifies how to [auto-scale the model](https://docs.ray.io/en/latest/serve/scaling-and-resource-allocation.html)
(via `autoscaling_config`) and what specific options you may need for your
Ray Actors during deployments (using `ray_actor_options`).

### Engine config

Engine is the abstraction for interacting with a model. It is responsible for scheduling and running the model inside a Ray Actor worker group.

The `engine_config` section specifies the Hugging Face model ID (`model_id`), how to initialize it and what parameters to use when generating tokens with an LLM.

RayLLM supports continuous batching, meaning incoming requests are processed as soon as they arrive, and can be added to batches that are already being processed. This means that the model is not slowed down by certain sentences taking longer to generate than others.

- `model_id` is the model ID. This is the ID that is used to refer to the model in the RayLLM API.
- `type` is the type of the engine. Currently that's only `VLLMEngine`.
- `generation` contains configuration related to default generation parameters.
- `hf_model_id` is the Hugging Face model ID. This can also be a path to a local directory. If not specified, defaults to `model_id`.
- `runtime_env` is a dictionary that contains Ray runtime environment configuration. It allows you to set per-model pip packages and environment variables. See [Ray documentation on Runtime Environments](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#runtime-environments) for more information.
- `s3_mirror_config` is a dictionary that contains configuration for loading the model from S3 instead of Hugging Face Hub. You can use this to speed up downloads.

### Scaling config

Finally, the `scaling_config` section specifies what resources should be used to serve the model - this corresponds to Ray AIR [ScalingConfig](https://docs.ray.io/en/latest/train/api/doc/ray.train.ScalingConfig.html). Note that the `scaling_config` applies to each model replica, and not the entire model deployment (in other words, each replica will have `num_workers` workers).
Notably, we use `resources_per_worker` to set [Ray custom resources](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html#id1)
to force the models onto specific node types - the corresponding resources are set in node definitions.

If you need to learn more about a specific configuration option, or need to add a new one, don't hesitate to reach out to the team.

## Adding a new model

To add an entirely new model to the zoo, you will need to create a new YAML file.
This file should follow the naming convention
`<organisation-name>--<model-name>-<model-parameters>-<extra-info>.yaml`. We recommend using one of the existing models as a template (ideally, one that is the same architecture as the model you are adding).

```yaml
# true by default - you can set it to false to ignore this model
# during loading
enabled: true
deployment_config:
  # This corresponds to Ray Serve settings, as generated with
  # `serve build`.
  autoscaling_config:
    min_replicas: 1
    initial_replicas: 1
    max_replicas: 8
    target_num_ongoing_requests_per_replica: 1.0
    metrics_interval_s: 10.0
    look_back_period_s: 30.0
    smoothing_factor: 1.0
    downscale_delay_s: 300.0
    upscale_delay_s: 90.0
  ray_actor_options:
    # Resources assigned to each model deployment. The deployment will be
    # initialized first, and then start prediction workers which actually hold the model.
    resources:
      accelerator_type_cpu: 0.01
engine_config:
  # Model id - this is a RayLLM id
  model_id: mosaicml/mpt-7b-instruct
  # Id of the model on Hugging Face Hub. Can also be a disk path. Defaults to model_id if not specified.
  hf_model_id: mosaicml/mpt-7b-instruct
  # vLLM keyword arguments passed when constructing the model.
  engine_kwargs:
    trust_remote_code: true
    # Optional quantization configuration. This is required only when serving an AWQ quantized model.
    quantization: awq
  # Optional Ray Runtime Environment configuration. See Ray documentation for more details.
  # Add dependent libraries, environment variables, etc.
  runtime_env:
    env_vars:
      YOUR_ENV_VAR: "your_value"
  # Optional configuration for loading the model from S3 instead of Hugging Face Hub. You can use this to speed up downloads or load models not on Hugging Face Hub.
  s3_mirror_config:
    bucket_uri: s3://large-dl-models-mirror/models--mosaicml--mpt-7b-instruct/main-safetensors/
  generation:
    # Prompt format to wrap queries in. {instruction} refers to user-supplied input.
    prompt_format:
      system: "{instruction}\n"  # System message. Will default to default_system_message
      assistant: "### Response:\n{instruction}\n"  # Past assistant message. Used in chat completions API.
      trailing_assistant: "### Response:\n"  # New assistant message. After this point, model will generate tokens.
      user: "### Instruction:\n{instruction}\n"  # User message.
      default_system_message: "Below is an instruction that describes a task. Write a response that appropriately completes the request."  # Default system message.
      system_in_user: false  # Whether the system prompt is inside the user prompt. If true, the user field should include '{system}'
      add_system_tags_even_if_message_is_empty: false  # Whether to include the system tags even if the user message is empty.
      strip_whitespace: false  # Whether to automaticall strip whitespace from left and right of user supplied messages for chat completions
    # Stopping sequences. The generation will stop when it encounters any of the sequences, or the tokenizer EOS token.
    # Those can be strings, integers (token ids) or lists of integers.
    # Stopping sequences supplied by the user in a request will be appended to this.
    stopping_sequences: ["### Response:", "### End"]

# Resources assigned to each model replica. This corresponds to Ray AIR ScalingConfig.
scaling_config:
  # If using multiple GPUs set num_gpus_per_worker to be 1 and then set num_workers to be the number of GPUs you want to use.
  num_workers: 1
  num_gpus_per_worker: 1
  num_cpus_per_worker: 4
  resources_per_worker:
    # You can use custom resources to specify the instance type / accelerator type
    # to use for the model.
    accelerator_type_a10: 0.01

```

### Adding a private model

To add a private model, you can either choose to use a filesystem path or an S3 mirror.

- For loading a model from file system, set `engine_config.hf_model_id` to an absolute filesystem path accessible from every node in the cluster and set `engine_config.model_id` to any ID you desire in the `organization/model` format, eg. `myorganization/llama2-finetuned`.
- For loading a model from S3, set `engine_config.s3_mirror_config.bucket_uri` to point to a folder containing your model and tokenizer files (`config.json`, `tokenizer_config.json`, `.bin`/`.safetensors` files, etc.) and set `engine_config.model_id` to any ID you desire in the `organization/model` format, eg. `myorganization/llama2-finetuned`. The model will be downloaded to a folder in the `<TRANSFORMERS_CACHE>/models--<organization-name>--<model-name>/snapshots/<HASH>` directory on each node in the cluster. `<HASH>` will be determined by the contents of `hash` file in the S3 folder, or default to `0000000000000000000000000000000000000000`. See the [HuggingFace transformers documentation](https://huggingface.co/docs/transformers/main/en/installation#cache-setup).

For loading a model from your local file system:

```yaml
engine_config:
  model_id: YOUR_MODEL_NAME
  hf_model_id: YOUR_MODEL_LOCAL_PATH
```

For loading a model from S3:

```yaml
engine_config:
  model_id: YOUR_MODEL_NAME
  s3_mirror_config:
    bucket_uri: s3://YOUR_BUCKET_NAME/YOUR_MODEL_FOLDER

```
