import sys

config_file = "/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/config.py"
model_file = "/usr/local/lib/python3.12/dist-packages/vllm/config/model.py"
tokenizer_file = "/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/tokenizer.py"

# Patch 1: transformers_utils/config.py
# Default rope_type to "default" instead of raising for unknown formats (Gemma 4)
with open(config_file) as f:
    content = f.read()

old = (
    '    if "rope_type" not in rope_scaling:\n'
    '        raise ValueError("rope_scaling should have a \'rope_type\' key")'
)
new = (
    '    if "rope_type" not in rope_scaling:\n'
    '        rope_scaling["rope_type"] = "default"  # patched: Gemma 4 compat'
)

if old in content:
    content = content.replace(old, new)
    with open(config_file, "w") as f:
        f.write(content)
    print(f"[OK] Patched {config_file}")
else:
    print(f"[WARN] Pattern not found in {config_file}", file=sys.stderr)
    sys.exit(1)

# Patch 2: config/model.py
# Add gemma4 to the list of model types that skip rope_scaling factor scaling
with open(model_file) as f:
    content = f.read()

old2 = 'if rope_scaling is not None and "gemma3" not in hf_config.model_type:'
new2 = 'if rope_scaling is not None and "gemma3" not in hf_config.model_type and "gemma4" not in hf_config.model_type:'

if old2 in content:
    content = content.replace(old2, new2)
    with open(model_file, "w") as f:
        f.write(content)
    print(f"[OK] Patched {model_file}")
else:
    print(f"[WARN] Pattern not found in {model_file}", file=sys.stderr)
    sys.exit(1)

# Patch 3: transformers_utils/tokenizer.py
# Handle missing all_special_tokens_extended attribute (removed in newer transformers)
with open(tokenizer_file) as f:
    content = f.read()

old3 = (
    '    tokenizer_all_special_tokens_extended = (\n'
    '        tokenizer.all_special_tokens_extended)'
)
new3 = (
    '    tokenizer_all_special_tokens_extended = (\n'
    '        getattr(tokenizer, "all_special_tokens_extended", tokenizer.all_special_tokens))'
)

if old3 in content:
    content = content.replace(old3, new3)
    with open(tokenizer_file, "w") as f:
        f.write(content)
    print(f"[OK] Patched {tokenizer_file}")
else:
    print(f"[WARN] Pattern not found in {tokenizer_file}", file=sys.stderr)
    sys.exit(1)

# Patch 4: model_executor/models/utils.py
# Skip unknown checkpoint weights (like audio_tower input_max quantization metadata)
# instead of raising ValueError — matches PyTorch strict=False behavior
utils_file = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/utils.py"

with open(utils_file) as f:
    content = f.read()

old4 = (
    '                msg = (f"There is no module or parameter named \'{prefix}\' "\n'
    '                       f"in {type(self.module).__name__}")\n'
    '                raise ValueError(msg)'
)
new4 = (
    '                msg = (f"There is no module or parameter named \'{prefix}\' "\n'
    '                       f"in {type(self.module).__name__}")\n'
    '                import sys as _sys; print(f"[WARN] Skipping unknown weight: {prefix}", file=_sys.stderr)  # patched: Gemma 4 compat\n'
    '                continue'
)

if old4 in content:
    content = content.replace(old4, new4)
    with open(utils_file, "w") as f:
        f.write(content)
    print(f"[OK] Patched {utils_file}")
else:
    print(f"[WARN] Pattern not found in {utils_file}", file=sys.stderr)
    sys.exit(1)

# Patch 5: v1/worker/utils.py
# Handle Gemma 4 multimodal encoder returning BaseModelOutputWithPast
# instead of the expected list/tuple/tensor format — extract last_hidden_state
utils_worker_file = "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/utils.py"

with open(utils_worker_file) as f:
    content = f.read()

old5 = (
    '    assert isinstance(mm_embeddings, (list, tuple, torch.Tensor)), (\n'
    '        "Expected multimodal embeddings to be a list/tuple of 2D tensors, "\n'
    '        f"or a single 3D tensor, but got {type(mm_embeddings)} "\n'
    '        "instead. This is most likely due to incorrect implementation "\n'
    '        "of the model\'s `get_multimodal_embeddings` method.")'
)
new5 = (
    '    if hasattr(mm_embeddings, "last_hidden_state"):  # patched: Gemma 4 compat\n'
    '        hs = mm_embeddings.last_hidden_state  # shape: (total_tokens, D)\n'
    '        mm_embeddings = list(hs.chunk(expected_num_items, dim=0))  # split into per-item 2D tensors\n'
    '    assert isinstance(mm_embeddings, (list, tuple, torch.Tensor)), (\n'
    '        "Expected multimodal embeddings to be a list/tuple of 2D tensors, "\n'
    '        f"or a single 3D tensor, but got {type(mm_embeddings)} "\n'
    '        "instead. This is most likely due to incorrect implementation "\n'
    '        "of the model\'s `get_multimodal_embeddings` method.")'
)

if old5 in content:
    content = content.replace(old5, new5)
    with open(utils_worker_file, "w") as f:
        f.write(content)
    print(f"[OK] Patched {utils_worker_file}")
else:
    print(f"[WARN] Pattern not found in {utils_worker_file}", file=sys.stderr)
    sys.exit(1)

# Patch 6: transformers/models/gemma4/modeling_gemma4.py
# Replace catastrophically memory-intensive reverse embedding lookup
# (creates (batch, seq, 262144, 2560) tensor → OOM) with a zero fallback
# when input_ids is None (as vLLM passes for multimodal inputs)
gemma4_file = "/usr/local/lib/python3.12/dist-packages/transformers/models/gemma4/modeling_gemma4.py"

with open(gemma4_file) as f:
    content = f.read()

old6 = (
    '        if input_ids is None:\n'
    '            with torch.no_grad():\n'
    '                input_ids = (\n'
    '                    (\n'
    '                        inputs_embeds[:, :, None, :]\n'
    '                        == self.embed_tokens.weight[None, None, :, :] * self.config.hidden_size**0.5\n'
    '                    )\n'
    '                    .all(dim=3)\n'
    '                    .nonzero()[:, 2]\n'
    '                )\n'
    '                try:\n'
    '                    input_ids = input_ids.view(inputs_embeds.shape[:2])\n'
    '                except RuntimeError:\n'
    '                    raise RuntimeError(\n'
    '                        "It seems like you tried to call `forward` from `inputs_embeds` without providing `input_ids`, and that "\n'
    '                        "the `inputs_embeds` you provided do not exactly match the embedding weights. Since Gemma4 needs to reverse "\n'
    '                        "the embedding to compute another embedding, make sure you provide exact `inputs_embeds`"\n'
    '                    )'
)
new6 = (
    '        if input_ids is None:\n'
    '            # patched: chunked argmax reverse embedding — avoids OOM from full (B,T,V,D) broadcast\n'
    '            # Uses 512-token chunks: each matmul is (1, 512, 262144) ~512MB, then freed\n'
    '            with torch.no_grad():\n'
    '                scale = self.config.hidden_size ** 0.5\n'
    '                B, T, D = inputs_embeds.shape\n'
    '                weights = self.embed_tokens.weight  # (V, D)\n'
    '                chunk_size = 512\n'
    '                chunks = []\n'
    '                for start in range(0, T, chunk_size):\n'
    '                    chunk = inputs_embeds[:, start:start + chunk_size, :] / scale  # (B, chunk, D)\n'
    '                    dots = torch.matmul(chunk, weights.T)  # (B, chunk, V)\n'
    '                    chunks.append(torch.argmax(dots, dim=-1))  # (B, chunk)\n'
    '                input_ids = torch.cat(chunks, dim=1)  # (B, T)'
)

if old6 in content:
    content = content.replace(old6, new6)
    with open(gemma4_file, "w") as f:
        f.write(content)
    print(f"[OK] Patched {gemma4_file}")
else:
    print(f"[WARN] Pattern not found in {gemma4_file}", file=sys.stderr)
    sys.exit(1)

print("All patches applied successfully.")
