# Aviary - Study stochastic parrots in the wild

Go on bird watch right now: [🦜🔍 Aviary 🦜🔍](http://aviary.anyscale.com/)

Aviary helps you deploy large language models (LLMs) with state-of-the-art optimizations on top of Ray Serve. 
In particular, it offers good support for Transformer models hosted on [Hugging Face](http://hf.co) and in many cases also 
supports [DeepSpeed](https://www.deepspeed.ai/) inference acceleration as well as continuous batching and paged attention. 
This template uses the `anyscale/aviary:latest-tgi` docker image.

In this guide, we will go over deploying a model locally using `serve run` and on an Anyscale Service. Alternatively, you can use the [Aviary CLI](https://github.com/ray-project/aviary/tree/master#using-the-aviary-cli) on this workspace. The CLI can help you compare the outputs of different models directly, rank them by quality, get a cost and latency estimate, and more. 

# Deploy a Large Language Model 
From the terminal use the Ray Serve CLI to deploy a model:

```shell
# Deploy the LightGPT model. 
serve run template/serve.yaml
```

The serve YAML file runs the lightgpt model. You can modify it to deploy any model in the `models` directory of this repo, provided you have the right GPU resources. You can also define your own model YAML file in the `models/` directory and run that instead. Follow the Aviary Model Registry [guide](models/README.md) for that.

### Query the model

Run the following command in a separate terminal. 

```shell
python template/request.py
```
```text
Output:
To make fried rice, start by heating up some oil in a large pan over medium-high
heat. Once the oil is hot, add your desired amount of vegetables and/or meat to the
pan. Cook until they are lightly browned, stirring occasionally. Add any other
desired ingredients such as eggs, cheese, or sauce to the pan. Finally, stir
everything together and cook for another few minutes until all the ingredients are
cooked through. Serve with your favorite sides and enjoy!
```

# Deploying on Anyscale Service

To deploy an application with one model on an Anyscale Service you can run:

```shell
anyscale service rollout -f template/service.yaml --name {ENTER_NAME_FOR_SERVICE_HERE}
```

This is setup to run the amazon/LightGPT model, but can be easily modified to run any of the other models in this repo.
In order to query the endpoint, you can modify the `template/request.py` script, replacing the query url with the Service URL found in the Service UI.


# Aviary Model Registry

Aviary allows you to easily add new models by adding a single configuration file.
To learn more about how to customize or add new models, 
see the [Aviary Model Registry](models/README.md).

# Getting Help and Filing Bugs / Feature Requests

We are eager to help you get started with Aviary. You can get help on: 

- Via Slack -- fill in [this form](https://docs.google.com/forms/d/e/1FAIpQLSfAcoiLCHOguOm8e7Jnn-JJdZaCxPGjgVCvFijHB5PLaQLeig/viewform) to sign up. 
- Via [Discuss](https://discuss.ray.io/c/llms-generative-ai/27). 

For bugs or for feature requests, please submit them [here](https://github.com/ray-project/aviary/issues/new).

We have people in both US and European time zones who will help answer your questions. 

