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
with an LLM (`generation`). We use Hugging Face Transformers under the hood.
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
    resources:
      instance_type_m5: 0.01

model_config:
  # Hugging Face model id
  model_id: mosaicml/mpt-7b-instruct
  initialization:
    # Optional runtime environment configuration. 
    # Add dependent libraries
    runtime_env:
      pip:
        - deepspeed==0.9.2
    # Optional configuration for loading the model from S3 instead of
    # Hugging Face Hub. You can use this to speed up downloads.
    s3_mirror_config:
      bucket_uri: s3://large-dl-models-mirror/models--mosaicml--mpt-7b-instruct/main-safetensors/
    # How to initialize the model.
    initializer:
      # Initializer type. Can be one of:
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
    # Leave as default for now.
    pipeline: default
  generation:
    # Max batch size to use when generating tokens
    max_batch_size: 22
    # Kwargs passed to `model.generate`
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

# Resources assigned to the model. This corresponds to Ray AIR ScalingConfig.
scaling_config:
  # DeepSpeed requires one worker per GPU - keep num_gpus_per_worker at 1 and
  # change num_workers.
  # For other initializers, you should set num_workers to 1 and instead change
  # num_gpus_per_worker.
  num_workers: 1
  num_gpus_per_worker: 1
  num_cpus_per_worker: 4
  resources_per_worker:
    # You can use custom resources to specify the instance type / accelerator type
    # to use for the model.
    instance_type_g5: 0.01
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
  generation:
    prompt_format: "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction:\n{instruction}\n### Response:\n"
    stopping_sequences: ["### Response:", "### End"]
```
