from setuptools import find_packages, setup

setup(
    name="aviary",
    version="0.0.1",
    description="A tool to deploy and query LLMs",
    packages=find_packages(include="aviary*"),
    include_package_data=True,
    package_data={"aviary": ["models/*"]},
    entry_points={
        "console_scripts": [
            "aviary=aviary.api.cli:app",
        ]
    },
    install_requires=["typer>=0.9", "rich"],
    extras_require={
        # TODO(tchordia): test whether this works, and determine how we can keep requirements
        # in sync
        "backend": [
            "async_timeout",
            "torch>=2.0.0",
            "torchaudio>=2.0.0",
            "torchvision>=0.15.2",
            "diffusers @ git+https://github.com/huggingface/diffusers.git",
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
            "pytorch-lightning",
            "ninja",
            "protobuf<3.21.0",
            "optimum @ git+https://github.com/huggingface/optimum.git",
            "torchmetrics",
            "lm_dataformat @ git+https://github.com/EleutherAI/lm_dataformat.git@4eec05349977071bf67fc072290b95e31c8dd836",
            "lm_eval==0.3.0",
            "tiktoken==0.1.2",
            "pybind11==2.6.2",
            "einops==0.3.0",
            "safetensors",
            "pydantic==1.10.7",
            "markdown-it-py[plugins]",
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
    },
    dependency_links=["https://download.pytorch.org/whl/cu118"],
    python_requires=">=3.8",
)
