# Aviary model registry

This is where all the stochastic parrots of the Aviary live.
Each model is defined by a YAML configuration file in this directory.

## Modify an existing model

To modify an existing model, simply edit the YAML file for that model.
Each config file consists of three sections: 

- `deployment_config`, 
- `model_config`, 
- and `scaling_config`.

It's probably best to check out examples of existing models to see how they are configured.

To give you a brief overview, the `deployment_config` section corresponds to
[Ray Serve configuration](https://docs.ray.io/en/latest/serve/production-guide/config.html)
and specifies how to [auto-scale the model](https://docs.ray.io/en/latest/serve/scaling-and-resource-allocation.html)
(via `autoscaling_config`) and what specific options you may need for your
Ray Actors during deployments (using `ray_actor_options`).

The `model_config` section specifies the Hugging Face model ID (`model_id`), how to 
initialize it (`initialization`) and what parameters to use when generating tokens
with an LLM (`generation`).

Aviary supports two different batching modes - static and continuous. In
static batching, incoming requests are batched together and sent to the model.
The model can only process the next batch once the current one is fully finished
(meaning that the batch takes as long to process as the single longest sentence).
In continuous batching, incoming requests are processed as soon as they arrive,
and can be added to batches that are already being processed. This means that
the model is not slowed down by certain sentences taking longer to generate than others.

### Static batching

For static batching, we use Hugging Face Transformers under the hood.

Aviary implements several initializer types:
- SingleDevice - just load the model onto a single GPU,
- DeviceMap - use the `device_map` argument to load the model onto multiple
  GPUs on a single node,
- DeepSpeed - use DeepSpeed to load the model onto multiple GPUs on a single
  or multiple nodes and run the model in tensor parallel mode (`deepspeed.init_inference`).
- LlamaCpp - use [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) to
  load the model. llama.cpp is separate from Torch & Hugging Face Transformers and uses its own model format.
  The model files are still downloaded from Hugging Face Hub - specify `model_filename` to control which
  file in the repository will be loaded.

Under the `generation` section, you also need to specify the `max_batch_size` and optionally `batch_wait_timeout_s` to configure static batching.

### Continuous batching

For continuous batching, Aviary uses Hugging Face text-generation-inference.
This is an optional requirement that has to be installed separately. Because the installation involves compilation from source, we recommend using our Docker image (`anyscale/aviary:latest-tgi`). You can see examples of continuous batching model configurations in the `models/tgi` folder.

The following settings can be configured under the `generation` section:
- `max_batch_total_tokens` - the maximum number of tokens that can be processed by the model. For `max_batch_total_tokens=1000`, you could fit `10` queries of `total_tokens=100` or a single query of `1000` tokens. Setting this too high will result in out-of-memory errors while setting it too low will result in underutilization of memory. Overall this number should be the largest possible amount that fits the remaining memory (after the model is loaded). In the future, we will add automatic tuning of this parameter.
- `max_total_tokens` - the maximum number of input+output tokens in a single request.
- `max_waiting_tokens` - defines how many tokens can be passed before forcing the waiting queries to be put on the batch (if the size of the batch allows for it). Usually, you shouldn't need to change it.
- `max_input_length` - maximum number of input tokens in a single request.
- `max_batch_prefill_tokens` - limits the number of tokens for the prefill operation.
- `waiting_served_ratio` - represents the ratio of waiting queries vs running queries where you want to start considering pausing the running queries to include the waiting ones into the same batch.

### Scaling config

Finally, the `scaling_config` section specifies what resources should be used to
serve the model - this corresponds to Ray AIR [ScalingConfig](https://docs.ray.io/en/latest/ray-air/api/doc/ray.air.ScalingConfig.html#ray-air-scalingconfig).
Notably, we use `resources_per_worker` to set [Ray custom resources](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html#id1)
to force the models onto specific node types - the corresponding resources are set
in node definitions.

If you need to learn more about a specific configuration option, or need to add a new
one, don't hesitate to reach out to the team.

## Adding a new model

To add an entirely new model to the zoo, you will need to create a new YAML file.
This file should follow the naming convention 
`<organisation-name>--<model-name>-<model-parameters>-<extra-info>.yaml`.

For instance, the YAML example shown next is stored in a file called
`mosaicml--mpt-7b-instruct.yaml`:

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

model_config:
  # Model id - this is an Aviary id
  model_id: mosaicml/mpt-7b-instruct
  # Batching type - static or continuous. Different
  # batching types support different initializers, pipelines and arguments.
  batching: static
  initialization:
    # Id of the model on Hugging Face Hub. Can also be a disk path. Defaults to model_id
    # if not specified.
    hf_model_id: mosaicml/mpt-7b-instruct
    # Optional runtime environment configuration. 
    # Add dependent libraries
    runtime_env:
      pip:
        - deepspeed==0.9.2
    # Optional configuration for loading the model from S3 instead of
    # Hugging Face Hub. You can use this to speed up downloads.
    s3_mirror_config:
      bucket_uri: s3://large-dl-models-mirror/models--mosaicml--mpt-7b-instruct/main-safetensors/
      s3_sync_args:
        - "--no-sign-request"
    # How to initialize the model.
    initializer:
      # Initializer type. For static batching, can be one of:
      # - SingleDevice - just load the model onto a single GPU
      # - DeviceMap - use the `device_map` argument to load the model onto multiple
      #   GPUs on a single node
      # - DeepSpeed - use DeepSpeed to load the model onto multiple GPUs on a single
      #   or multiple nodes and run the model in tensor parallel mode (`deepspeed.init_inference`)
      type: SingleDevice
      # dtype to use when loading the model
      dtype: bfloat16
      # kwargs to pass to `AutoModel.from_pretrained`
      from_pretrained_kwargs:
        trust_remote_code: true
        use_cache: true
      # Whether to use Hugging Face Optimum BetterTransformer to inject flash attention
      # (may not work with all models)
      use_bettertransformer: false
      # Whether to use Torch 2.0 `torch.compile` to compile the model
      torch_compile:
        backend: inductor
        mode: max-autotune
    # Aviary pipeline class. This is separate from Hugging Face pipelines.
    # Leave as transformers for now.
    pipeline: transformers
  generation:
    # Max batch size to use when generating tokens
    max_batch_size: 22
    # Default kwargs passed to `model.generate`
    generate_kwargs:
      do_sample: true
      max_new_tokens: 512
      min_new_tokens: 16
      top_p: 1.0
      top_k: 0
      temperature: 0.1
      repetition_penalty: 1.1
    # Prompt format to wrap queries in. Must be empty or contain `{instruction}`.
    prompt_format: "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction:\n{instruction}\n### Response:\n"
    # Stopping sequences. The generation will stop when it encounters any of the sequences, or the tokenizer EOS token.
    # Those can be strings, integers (token ids) or lists of integers.
    stopping_sequences: ["### Response:", "### End"]

# Resources assigned to each model replica. This corresponds to Ray AIR ScalingConfig.
scaling_config:
  # DeepSpeed/TextGenerationInference requires one worker per GPU - keep num_gpus_per_worker at 1 and
  # change num_workers.
  # For other initializers, you should set num_workers to 1 and instead change
  # num_gpus_per_worker.
  num_workers: 1
  num_gpus_per_worker: 1
  num_cpus_per_worker: 4
  resources_per_worker:
    # You can use custom resources to specify the instance type / accelerator type
    # to use for the model.
    accelerator_type_a10: 0.01
```

You will notice that many models only deviate very slightly from each other.
For instance, the "chat" version of the above example, stored in 
`mosaicml--mpt-7b-chat.yaml` only has four values that differ from the above example:

```yaml
...
model_config:
  model_id: mosaicml/mpt-7b-chat
  initialization:
    s3_mirror_config:
      bucket_uri: s3://large-dl-models-mirror/models--mosaicml--mpt-7b-instruct/main-safetensors/
    s3_sync_args:
      - "--no-sign-request"
  generation:
    prompt_format: "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction:\n{instruction}\n### Response:\n"
    stopping_sequences: ["### Response:", "### End"]
```