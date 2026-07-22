ARG BASE_IMAGE=reg.docker.alibaba-inc.com/aisw/llm:v1.6.1-pytorch2.6.0-ubuntu22.04-cuda12.6-vllm0.7.3-py310
FROM ${BASE_IMAGE}

ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    LARA_ATTN_IMPLEMENTATION=eager

WORKDIR /workspace/Latent-gui-agent

COPY requirements-ppu.txt /tmp/requirements-ppu.txt
RUN python -m pip install --no-cache-dir --upgrade -r /tmp/requirements-ppu.txt

COPY . /workspace/Latent-gui-agent
RUN python -m pip install --no-cache-dir --no-deps --no-build-isolation -e . \
    && python -c "from transformers import Qwen3VLForConditionalGeneration; import qwen3_gui_agent; print('Qwen3-VL project import: PASS')" \
    && python -m pip check

CMD ["bash"]
