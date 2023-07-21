from setuptools import find_packages, setup

setup(
    name="aviary",
    version="0.1.1",
    description="A tool to deploy and query LLMs",
    packages=find_packages(include="aviary*"),
    include_package_data=True,
    package_data={"aviary": ["models/*"]},
    entry_points={
        "console_scripts": [
            "aviary=aviary.cli:app",
        ]
    },
    install_requires=[
        "typer>=0.9",
        "rich",
        "typing_extensions==4.5.0",
        "requests",
    ],
    extras_require={
        # TODO(tchordia): test whether this works, and determine how we can keep requirements
        # in sync
        "backend": [
            "async_timeout",
            "torch>=2.0.0",
            "torchaudio>=2.0.0",
            "torchvision>=0.15.2",
            "accelerate",
            "transformers>=4.25.1",
            "datasets",
            "ftfy",
            "tensorboard",
            "sentencepiece",
            "Jinja2",
            "numexpr>=2.7.3",
            "hf_transfer",
            "evaluate",
            "bitsandbytes",
            "deepspeed @ git+https://github.com/Yard1/DeepSpeed.git@aviary",
            "numpy<1.24",
            "ninja",
            "protobuf<3.21.0",
            "optimum @ git+https://github.com/huggingface/optimum.git",
            "torchmetrics",
            "safetensors",
            "pydantic==1.10.7",
            "einops",
            "markdown-it-py[plugins]",
            "fastapi-versioning",
        ],
        "frontend": [
            "gradio",
            "aiorwlock",
            "ray",
            "pymongo",
            "pandas",
            "boto3",
        ],
        "dev": [
            "pre-commit",
            "ruff==0.0.270",
            "black==23.3.0",
        ],
        "test": [
            "pytest",
        ],
        "docs": [
            "mkdocs-material",
        ],
    },
    dependency_links=["https://download.pytorch.org/whl/cu118"],
    python_requires=">=3.8",
)
