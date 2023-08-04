import requests
from aviary.backend.server.models import (
    Args,
    AviaryModelResponse,
    Prompt,
)


prompt = Prompt(prompt="What is the best place to eat in San Francisco?",
            parameters={
                "max_new_tokens": 128,
            },)

resp = requests.post(
    "http://127.0.0.1:8000/amazon--LightGPT/stream/", json={
                    "prompt": prompt.dict(),
                    "priority":0,
                }, stream = True
)
resp.raise_for_status()
import json
for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
    #print(chunk["generated_text"])
    token = json.loads(chunk)["generated_text"]
    if (token != None):
        print(token, end="", flush=True)

print("")
