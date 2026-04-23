FROM containers.renci.org/helxplatform/vllm:0.11.0

# Upgrade transformers for Gemma 4 support (vLLM stays at 0.11.0)
RUN pip install --no-cache-dir "transformers>=4.51.0" huggingface_hub

# Patch vLLM 0.11.0 to handle Gemma 4 rope_scaling format
COPY patch_vllm.py /tmp/patch_vllm.py
RUN python3 /tmp/patch_vllm.py
