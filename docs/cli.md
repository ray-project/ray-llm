# Using the Aviary CLI

Aviary comes with a CLI that allows you to interact with the backend directly, without
using the Gradio frontend.
Installing Aviary as described earlier will install the `aviary` CLI as well.
You can get a list of all available commands by running `aviary --help`.

Currently, `aviary` supports a few basic commands, all of which can be used with the
`--help` flag to get more information:

```shell
# Get a list of all available models in Aviary
aviary models

# Query a model with a list of prompts
aviary query --model <model-name> --prompt <prompt_1> --prompt <prompt_2>

# Run a query on a text file of prompts
aviary query  --model <model-name> --prompt-file <prompt-file>

# Run a query with streaming
aviary stream --model <model-name> --prompt <prompt_1>

# Evaluate the quality of responses with GPT-4 for evaluation
aviary evaluate --input-file <query-result-file>

# Start a new model in Aviary from provided configuration
aviary run <model>
```

## CLI examples

### Listing all available models

```shell
aviary models
```
```text
mosaicml/mpt-7b-instruct
meta-llama/Llama-2-7b-chat-hf
```

### Running two models on the same prompt

```shell
aviary query --model mosaicml/mpt-7b-instruct --model meta-llama/Llama-2-7b-chat-hf \
  --prompt "what is love?"
```
```text
mosaicml/mpt-7b-instruct:
love can be defined as feeling of affection, attraction or ...
meta-llama/Llama-2-7b-chat-hf:
Love is a feeling of strong affection and care for someone or something...
```

### Running a batch-query of two prompts on the same model

```shell
aviary query --model mosaicml/mpt-7b-instruct \
  --prompt "what is love?" --prompt "why are we here?"
```

### Running a query on a text file of prompts

```shell
aviary query --model mosaicml/mpt-7b-instruct --prompt-file prompts.txt
```

### Running a streaming response

```shell
aviary stream --model mosaicml/mpt-7b-instruct --prompt "What is love?"
```

### Evaluating the quality of responses with GPT-4 for evaluation

```shell
 aviary evaluate --input-file aviary-output.json --evaluator gpt-4
```

This will result in a leaderboard-like ranking of responses, but also save the
results to file. 

You can also use the Gradio API directly, by following the instructions
provided in the [Aviary documentation](https://aviary.anyscale.com/?view=api).
