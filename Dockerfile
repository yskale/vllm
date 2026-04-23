FROM containers.renci.org/helxplatform/vllm:0.11.0

# Upgrade transformers using the same python3 that vLLM uses, pin numpy for numba compat
RUN python3 -m pip install --no-cache-dir --force-reinstall "transformers>=4.51.0" huggingface_hub "numpy<2.3" "mistral-common"

# Verify the upgrade took effect for the correct python3
RUN python3 -c "import transformers; print('transformers version:', transformers.__version__); assert tuple(int(x) for x in transformers.__version__.split('.')[:2]) >= (4, 51), 'transformers version too old'"

# Patch vLLM 0.11.0 to handle Gemma 4 rope_scaling format
COPY patch_vllm.py /tmp/patch_vllm.py
RUN python3 /tmp/patch_vllm.py
