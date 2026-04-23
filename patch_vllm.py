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

print("All patches applied successfully.")
