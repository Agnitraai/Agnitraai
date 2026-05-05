# Pinned environment for the Llama-3-8B / H100 benchmark.
#
# Built on the official PyTorch CUDA image so the toolchain matches what
# vLLM and TensorRT-LLM expect. Do not switch to a slim image: vLLM
# requires CUDA libraries at runtime.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Install benchmark deps first so the layer is cached when only the
# Agnitra source changes.
COPY benchmarks/llama3_h100/requirements.txt /work/benchmarks/llama3_h100/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /work/benchmarks/llama3_h100/requirements.txt \
 && pip install hf_transfer

# Install Agnitra from the repo source.
COPY pyproject.toml setup.py README.md /work/
COPY agnitra /work/agnitra
RUN pip install -e .

# Bring in the benchmark scripts last (changes here invalidate the
# smallest layer).
COPY benchmarks /work/benchmarks

WORKDIR /work/benchmarks/llama3_h100

CMD ["./run.sh"]
