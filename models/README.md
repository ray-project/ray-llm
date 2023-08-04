# Aviary model registry

This is where all the stochastic parrots of the Aviary live.
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

Engine is the Aviary abstraction for interacting with a model. It is responsible for scheduling and running the model inside a Ray Actor worker group.

The `engine_config` section specifies the Hugging Face model ID (`model_id`), how to initialize it and what parameters to use when generating tokens with an LLM.

Aviary supports continuous batching, meaning incoming requests are processed as soon as they arrive, and can be added to batches that are already being processed. This means that the model is not slowed down by certain sentences taking longer to generate than others.


* `model_id` is the Aviary model ID. This is the ID that is used to refer to the model in the Aviary API.
* `type` is the type of the engine. Currently that's only `TextGenerationInferenceEngine`.
* `generation` contains configuration related to default generation parameters.
* `scheduler.policy` contains configuration related to the continuous batching scheduler. Those settings are very important for the performance and memory usage of the model.
* `hf_model_id` is the Hugging Face model ID. This can also be a path to a local directory. If not specified, defaults to `model_id`.
* `runtime_env` is a dictionary that contains Ray runtime environment configuration. It allows you to set per-model pip packages and environment variables. See [Ray documentation on Runtime Environments](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#runtime-environments) for more information.
* `s3_mirror_config` is a dictionary that contains configuration for loading the model from S3 instead of Hugging Face Hub. You can use this to speed up downloads.

#### Scheduler policy

The scheduler is responsible for batching incoming requests together before running
an iteration of inference on the model. It has a policy to determine how requests should
be batched together. Currently the only policy is a quota based policy that uses bin
packing to determine how many requests can fit in a single batch.

The following settings can be configured under the `scheduler.policy` section:
- `max_total_tokens` - the maximum number of input+output tokens in a single request. This should usually be set to the model context length.
- `max_input_length` - maximum number of input tokens in a single request. Must be less than or equal to `max_total_tokens`.
- `max_batch_prefill_tokens` - limits the number of tokens for the prefill operation. This is the main parameter to tune in regards to throughput and memory consumption. Setting this too high will result in out-of-memory errors while setting it too low will result in underutilization of memory.
- `max_batch_total_tokens` - limits the total number of tokens in a running batch. This parameter will be automatically derived from `max_batch_prefill_tokens` for most model architectures, but you may need to set it manually for other architectures (an exception will be thrown if this parameter is unspecified and the model requires it).
- `max_iterations_curr_batch` - defines how many scheduler iterations can be passed before forcing the waiting queries to be put on the batch (if the size of the batch allows for it). Usually, you shouldn't need to change it.
- `waiting_served_ratio` - represents the ratio of waiting queries vs running queries where you want to start considering pausing the running queries to include the waiting ones into the same batch.

### Scaling config

Finally, the `scaling_config` section specifies what resources should be used to serve the model - this corresponds to Ray AIR [ScalingConfig](https://docs.ray.io/en/latest/ray-air/api/doc/ray.air.ScalingConfig.html#ray-air-scalingconfig). Note that the `scaling_config` applies to each model replica, and not the entire model deployment (in other words, each replica will have `num_workers` workers).
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
  # Model id - this is an Aviary id
  model_id: mosaicml/mpt-7b-instruct
  # Id of the model on Hugging Face Hub. Can also be a disk path. Defaults to model_id if not specified.
  hf_model_id: mosaicml/mpt-7b-instruct
  # TGI and transformers keyword arguments passed when constructing the model.
  model_init_kwargs:
    trust_remote_code: true
  # This is a metadata field that is used to display information about the model in the UI.
  model_description: mosaic mpt 7b is a transformer trained from scratch...
  # This is a metadata field that is used to display information about the model in the UI.
  model_url: https://www.mosaicml.com/blog/mpt-7b
  # Optional Ray Runtime Environment configuration. See Ray documentation for more details.
  # Add dependent libraries, environment variables, etc.
  runtime_env:
    env_vars:
      YOUR_ENV_VAR: "your_value"
  # Optional configuration for loading the model from S3 instead of Hugging Face Hub. You can use this to speed up downloads or load models not on Hugging Face Hub.
  s3_mirror_config:
    bucket_uri: s3://large-dl-models-mirror/models--mosaicml--mpt-7b-instruct/main-safetensors/
  # Parameters for configuring the scheduler. See above for more details.
  scheduler:
    policy:
      max_batch_prefill_tokens: 58000
      max_batch_total_tokens: 140000
      max_input_length: 2048
      max_iterations_curr_batch: 20
      max_total_tokens: 4096
      type: QuotaBasedTaskSelectionPolicy
      waiting_served_ratio: 1.2
  generation:
    # Default kwargs passed to `model.generate`. These can be overrided by a
    # user's request.
    generate_kwargs:
      do_sample: true
      max_new_tokens: 512
      min_new_tokens: 16
      top_p: 1.0
      top_k: 0
      temperature: 0.1
      repetition_penalty: 1.1
    # Prompt format to wrap queries in. {instruction} refers to user-supplied input.
    prompt_format:
      system: "{instruction}\n"  # System message. Will default to default_system_message
      assistant: "### Response:\n{instruction}\n"  # Past assistant message. Used in chat completions API.
      trailing_assistant: "### Response:\n"  # New assistant message. After this point, model will generate tokens.
      user: "### Instruction:\n{instruction}\n"  # User message.
      default_system_message: "Below is an instruction that describes a task. Write a response that appropriately completes the request."  # Default system message.
    # Stopping sequences. The generation will stop when it encounters any of the sequences, or the tokenizer EOS token.
    # Those can be strings, integers (token ids) or lists of integers.
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
