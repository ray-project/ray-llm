## Using the Aviary CLI

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

# Evaluate the quality of responses with GPT-4 for evaluation
aviary evaluate --input-file <query-result-file>

# Start a new model in Aviary from provided configuration
aviary run <model>
```

### CLI examples

#### Listing all available models

```shell
aviary models
```
```text
mosaicml/mpt-7b-instruct
CarperAI/stable-vicuna-13b-delta
databricks/dolly-v2-12b
RWKV/rwkv-raven-14b
mosaicml/mpt-7b-chat
stabilityai/stablelm-tuned-alpha-7b
lmsys/vicuna-13b-delta-v1.1
mosaicml/mpt-7b-storywriter
h2oai/h2ogpt-oasst1-512-12b
OpenAssistant/oasst-sft-7-llama-30b-xor
```

#### Running two models on the same prompt

```shell
aviary query --model mosaicml/mpt-7b-instruct --model RWKV/rwkv-raven-14b \
  --prompt "what is love?"
```
```text
mosaicml/mpt-7b-instruct:
love can be defined as feeling of affection, attraction or ...
RWKV/rwkv-raven-14b:
Love is a feeling of strong affection and care for someone or something...
```

#### Running a batch-query of two prompts on the same model

```shell
aviary query --model mosaicml/mpt-7b-instruct \
  --prompt "what is love?" --prompt "why are we here?"
```

#### Running a query on a text file of prompts

```shell
aviary query --model mosaicml/mpt-7b-instruct --prompt-file prompts.txt
```

#### Evaluating the quality of responses with GPT-4 for evaluation

```shell
 aviary evaluate --input-file aviary-output.json --evaluator gpt-4
```

This will result in a leaderboard-like ranking of responses, but also save the
results to file:

```shell
What is the best indie band of the 90s?
                                              Evaluation results (higher ranks are better)                                               
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Model                    ┃ Rank ┃                                                                                            Response ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ mosaicml/mpt-7b-instruct │ 1    │  The Shins are often considered to be one of the greatest bands from this era, with their album 'Oh │
│                          │      │        Inverted World' being widely regarded as one of the most influential albums in recent memory │
│ RWKV/rwkv-raven-14b      │ 2    │ It's subjective and depends on personal taste. Some people might argue that Nirvana or The Smashing │
│                          │      │                       Pumpkins were the best, while others might prefer Sonic Youth or Dinosaur Jr. │
└──────────────────────────┴──────┴─────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

You can also use the Gradio API directly, by following the instructions
provided in the [Aviary documentation](https://aviary.anyscale.com/?view=api).
