from aviary import sdk
from aviary import cli
import os
os.environ["AVIARY_URL"] = "http://localhost:8000"
os.environ["AVIARY_TOKEN"] = ""

for chunk in sdk.stream("amazon--LightGPT", "What are the best restaurants in San Francisco?"):
    text = cli._get_text(chunk)
    cli.rp(text, end="")

cli.rp("")