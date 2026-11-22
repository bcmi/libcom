"""Strict loader for the artistic and photorealistic DiffSynth Qwen-Image-2.1 LoRAs."""

from pathlib import Path


TARGETS = (
    "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
    "img_mlp.proj", "img_mlp.out", "img_mlp.gate_layer",
)
EXPECTED_MODULES = {
    f"transformer_blocks.{block}.{target}"
    for block in range(32) for target in TARGETS
}
PREFIXES = ("module.", "_orig_mod.", "base_model.model.", "pipe.dit.", "dit.", "transformer.")


def convert_lora(raw):
    """Return Diffusers keys and reject every unexpected or missing tensor."""
    converted = {}
    for name, tensor in raw.items():
        key = name
        while True:
            prefix = next((prefix for prefix in PREFIXES if key.startswith(prefix)), None)
            if prefix is None:
                break
            key = key[len(prefix):]
        key = key.replace(".lora_A.default.weight", ".lora_A.weight")
        key = key.replace(".lora_B.default.weight", ".lora_B.weight")
        if not key.endswith((".lora_A.weight", ".lora_B.weight")):
            raise ValueError(f"Unexpected checkpoint tensor: {name}")
        full_key = f"transformer.{key}"
        if full_key in converted:
            raise ValueError(f"Duplicate LoRA tensor: {full_key}")
        converted[full_key] = tensor

    actual = set(converted)
    expected = {
        f"transformer.{module}.lora_{side}.weight"
        for module in EXPECTED_MODULES for side in "AB"
    }
    if actual != expected:
        raise ValueError(
            f"LoRA tensors do not match the trained 32-block adapter: "
            f"missing={sorted(expected - actual)[:8]}, unexpected={sorted(actual - expected)[:8]}"
        )
    for module in EXPECTED_MODULES:
        a = converted[f"transformer.{module}.lora_A.weight"]
        b = converted[f"transformer.{module}.lora_B.weight"]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != 32 or b.shape[1] != 32:
            raise ValueError(f"Expected rank-32 linear LoRA at {module}: A={tuple(a.shape)}, B={tuple(b.shape)}")
    return converted


def load_lora(pipeline, checkpoint, scale=1.0, adapter_name="qwen_edit"):
    from safetensors.torch import load_file

    path = Path(checkpoint).expanduser().resolve(strict=True)
    converted = convert_lora(load_file(str(path), device="cpu"))
    for module in EXPECTED_MODULES:
        layer = pipeline.transformer.get_submodule(module)
        a = converted[f"transformer.{module}.lora_A.weight"]
        b = converted[f"transformer.{module}.lora_B.weight"]
        if tuple(layer.weight.shape) != (b.shape[0], a.shape[1]):
            raise ValueError(f"LoRA/base model shape mismatch at {module}")

    adapter = adapter_name
    pipeline.load_lora_weights(converted, adapter_name=adapter)
    pipeline.set_adapters(adapter, adapter_weights=float(scale))
    for module in EXPECTED_MODULES:
        layer = pipeline.transformer.get_submodule(module)
        if adapter not in layer.lora_A or adapter not in layer.lora_B:
            raise RuntimeError(f"Diffusers did not attach LoRA at {module}")
    return {"checkpoint": str(path), "adapter_name": adapter,
            "modules": len(EXPECTED_MODULES), "tensors": len(converted),
            "rank": 32, "scale": float(scale)}
