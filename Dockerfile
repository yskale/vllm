FROM containers.renci.org/helxplatform/vllm:0.11.0

RUN pip install --no-cache-dir --upgrade "transformers>=4.51.0" "vllm" huggingface_hub
