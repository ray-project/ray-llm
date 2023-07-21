import requests

resp = requests.post(
    "http://127.0.0.1:8000/stream/", json={"prompt": "What is the best place to eat in San Francisco?","use_prompt_format":"true"}, stream = True
)

import json
resp.raise_for_status()
for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
    print(json.loads(chunk)["generated_text"], end="", flush=True)

print("")
