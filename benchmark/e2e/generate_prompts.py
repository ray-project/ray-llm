#!/usr/bin/env python3

import asyncio
import json
import os
import random
import sys

import aiohttp
import requests
from transformers import AutoTokenizer


async def generate_fixed_length_prompt(input_prompt, num_tokens_per_prompt):
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b")
    output_prompt = input_prompt
    tries = 0
    max_tries = 25

    async with aiohttp.ClientSession() as session:
        while True:
            max_new_tokens = num_tokens_per_prompt - len(
                tokenizer.encode(output_prompt)
            )

            if max_new_tokens == 0:
                break

            if tries > max_tries:
                raise ValueError("max tries")

            print(
                f"Missing {max_new_tokens} to get to {num_tokens_per_prompt}",
                file=sys.stderr,
            )

            tries += 1

            generate_input = dict(
                inputs=input_prompt,
                parameters=dict(
                    max_new_tokens=max_new_tokens,
                ),
            )

            async with session.post(
                "http://localhost:3000/generate", json=generate_input
            ) as resp:
                response = await resp.json()
            output_prompt = output_prompt + response["generated_text"]

    return output_prompt


async def main():
    if not os.path.exists("words"):
        resp = requests.get("https://www.mit.edu/~ecprice/wordlist.10000")  # noqa
        with open("words", "w") as f:  # noqa
            f.write(resp.text)  # noqa

    with open("words", "r") as f:  # noqa
        words = [word.strip() for word in f.readlines()]
    num_prompts = 20
    desired_length = 1024
    prompt_seeds = random.sample(words, k=num_prompts)
    print(f"Prompt seeds {prompt_seeds}", file=sys.stderr)

    outputs = [
        generate_fixed_length_prompt(f"Tell me about: {seed}", desired_length)
        for seed in prompt_seeds
    ]
    outputs = await asyncio.gather(*outputs, return_exceptions=True)

    non_except_outputs = 0
    for output in outputs:
        if isinstance(output, Exception):
            continue
        non_except_outputs += 1
        print(json.dumps(output))

    print(
        f"Generated {non_except_outputs} outputs of length {desired_length}",
        file=sys.stderr,
    )


asyncio.run(main())
