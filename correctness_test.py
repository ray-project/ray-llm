import os
import time

import ray

import aviary.api.sdk

model_name = "lmsys/vicuna-13b-delta-v1_1"

# model_name = "OpenAssistant/falcon-7b-sft-top1-696"

"""
I'm a software engineer with a passion for building elegant and efficient solutions to complex problems. I enjoy working with a variety of technologies and programming languages, and I'm always eager to learn new things.

In my free time, I enjoy hiking, playing guitar, and reading about science and philosophy. I'm also a big fan of dogs – I have two of my own!

I'm excited to share my knowledge and experience with you, and I hope you find my content helpful and informative. If you have any questions or comments, feel free to reach out to me!"""


# Make sure the following kwargs are set in yaml
# generate_kwargs:
#   do_sample: false
#   max_new_tokens: 512
#   min_new_tokens: 16
#   temperature: 1
#   top_p: 1.0
#   repetition_penalty: 1
#   return_token_type_ids: false
#   ignore_eos_token: false
@ray.remote(num_cpus=0.01)
def query(prompt):
    print("querying")
    os.environ["AVIARY_URL"] = "http://localhost:8000"
    return aviary.api.sdk.query(model_name, prompt)["generated_text"]


def run_test():
    output = []
    for _i in range(1):
        output.extend(ray.get([query.remote("Hello, world!") for _ in range(10)]))
        time.sleep(1)
    print(len(set(output)))
    with open("output_1x10.txt", "w") as f:
        f.write("\n".join(output))


for _i in range(10):
    run_test()
