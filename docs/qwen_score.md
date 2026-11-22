# Composition Quality Assessment with QwenScoreModel

QwenScoreModel evaluates the composition quality of a composite image with a vision-language model (Qwen3-VL-8B) fine-tuned with task-specific LoRA adapters. It covers three assessment dimensions in a unified interface:

> **QwenScoreModel** (Qwen3-VL + LoRA adapters)<br>
>

## Brief Method Summary

Given a composite image and its foreground mask, the model is prompted to judge one quality dimension and answer with a single digit (1 for reasonable/harmonious, 0 for not). The score is derived from the softmax probability of the answer token `1` over the digits `0`/`1`, thus ranging from 0 to 1, where a larger score indicates better quality. The scoring protocol is identical to the training/evaluation protocol, so the scores are directly comparable across the three tasks.

Three `model_type` options are supported, each backed by its own LoRA adapter:

| `model_type` | Task | Default LoRA | Default `max_pixels` |
|---|---|---|---|
| `QwenVL-OPA` | Object placement assessment | `lora_opa_qwen3_4k_highres` | 802816 |
| `QwenVL-FOS` | Foreground-background compatibility | `lora_fos_qwen3_4k_highres` | 802816 |
| `QwenVL-Harmony` | Illumination/color tone consistency | `lora_harmony_highres` | 802816 |

**Test-time augmentation (TTA)** is enabled by default: each task aggregates 3 forward passes with the view set that performed best on the validation set (`orig + hflip + 1.1x zoom` with mean aggregation for OPA/FOS; `orig + 1.1x + 1.2x zoom` with median aggregation for Harmony, since flipping changes the light direction). Pass `enable_tta=False` for a single forward pass.

## Getting Started

```python
from libcom import QwenScoreModel

composite_image = 'composite.png'
composite_mask  = 'composite_mask.png'   # binary foreground mask

for model_type in ['QwenVL-OPA', 'QwenVL-FOS', 'QwenVL-Harmony']:
    net = QwenScoreModel(device=0, model_type=model_type)
    score = net(composite_image, composite_mask)
    print(model_type, f'{score:.4f}')
```

## Model Weights

The base model and LoRA adapters are not bundled with the package. Place them in one of the following locations (checked in order):

1. Constructor arguments: `base_model=...` / `lora_path=...`
2. Environment variables: `LIBCOM_QWEN_BASE_MODEL` (base), `LIBCOM_QWEN_LORA_DIR` (LoRA root)
3. `libcom/qwen_score/pretrained_models/` inside the package
4. `$LIBCOM_QWEN_CKPT_DIR/<name>` (default checkpoint root)

```
<LIBCOM_QWEN_CKPT_DIR>
├── Qwen3-VL-8B-Instruct/         # base model (HuggingFace format)
├── lora_opa_qwen3_4k_highres/    # OPA adapter
├── lora_fos_qwen3_4k_highres/    # FOS adapter
└── lora_harmony_highres/         # Harmony adapter
```

If none of the local paths exist, the base model falls back to the official HuggingFace id `Qwen/Qwen3-VL-8B-Instruct` (about 16GB, downloaded automatically on first use). Missing LoRA adapters are downloaded automatically through the libcom pretrained-model mechanism (`BCMIZB/Libcom_pretrained_models`, packaged as `<lora_name>.zip`).

## Performance Comments

- The Qwen3-VL-8B base model is loaded in bf16 and requires about **16~18GB GPU memory**; a dedicated GPU is recommended.
- Reference metrics (validation set, single view): OPA AUC 0.9314 / Acc 0.8475; FOS AUC 0.7767; Harmony AUC 0.7911. TTA further improves Harmony Acc from 0.6951 to 0.7309.
- All three tasks share `max_pixels=802816`, consistent with the adapter training setting, so the three scores are comparable.
