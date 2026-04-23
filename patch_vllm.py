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
    '            # patched: recover input_ids saved by gpu_model_runner before it cleared them.\n'
    '            # vLLM v1 pre-computes inputs_embeds from input_ids (gpu_model_runner.py), then\n'
    '            # sets input_ids=None before calling the model.  We save them in a module-level\n'
    '            # buffer (_LAST_INPUT_IDS_FOR_PLE) via Patch 7 below so PLE can use the real IDs.\n'
    '            # Fallback to None (context-only PLE) if the buffer is absent.\n'
    '            try:\n'
    '                import vllm.v1.worker.gpu_model_runner as _gmr_mod\n'
    '                saved = getattr(_gmr_mod, "_LAST_INPUT_IDS_FOR_PLE", None)\n'
    '            except Exception:\n'
    '                saved = None\n'
    '            if saved is not None:\n'
    '                input_ids = saved.unsqueeze(0) if saved.dim() == 1 else saved\n'
    '                # Pad to match inputs_embeds seq length (non-zero only when DP adds padding)\n'
    '                if inputs_embeds is not None and input_ids.shape[-1] < inputs_embeds.shape[-2]:\n'
    '                    pad_len = inputs_embeds.shape[-2] - input_ids.shape[-1]\n'
    '                    input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=0)\n'
    '            else:\n'
    '                return None  # context-only PLE fallback'
)

if old6 in content:
    content = content.replace(old6, new6)
    with open(gemma4_file, "w") as f:
        f.write(content)
    print(f"[OK] Patched {gemma4_file}")
else:
    print(f"[WARN] Pattern not found in {gemma4_file}", file=sys.stderr)
    sys.exit(1)

# Patch 7: v1/worker/gpu_model_runner.py
# Save original input_ids to a module-level buffer BEFORE clearing them.
# Gemma4's get_per_layer_inputs reads this buffer when input_ids=None (Patch 6 above).
gpu_runner_file = "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py"

with open(gpu_runner_file) as f:
    content = f.read()

old7 = (
    '            input_ids = None\n'
    '            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]\n'
    '            model_kwargs = {'
)
new7 = (
    '            # patched: save input_ids for Gemma4 PLE before clearing (Patch 7 / Patch 6)\n'
    '            import vllm.v1.worker.gpu_model_runner as _gmr_self_mod\n'
    '            _gmr_self_mod._LAST_INPUT_IDS_FOR_PLE = self.input_ids.gpu[:num_scheduled_tokens].clone()\n'
    '            input_ids = None\n'
    '            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]\n'
    '            model_kwargs = {'
)

if old7 in content:
    content = content.replace(old7, new7)
    with open(gpu_runner_file, "w") as f:
        f.write(content)
    print(f"[OK] Patched {gpu_runner_file}")
else:
    print(f"[WARN] Pattern not found in {gpu_runner_file}", file=sys.stderr)
    sys.exit(1)

print("All patches applied successfully.")
