# Generative Composition with QwenEditModel

QwenEditModel is a generative image composition model based on **Qwen-Image-2.1** with mode-specific LoRA adapters. It inserts the foreground object into the specified bounding box of the background, harmonizes/stylizes the object to match the background, and generates shadow/reflection accordingly.

> **QwenEditModel** (Qwen-Image-2.1 + DiffSynth LoRA adapters)<br>
>

## Brief Method Summary

Two `model_type` options are supported, each backed by its own LoRA adapter:

| `model_type` | Style | Default LoRA |
|---|---|---|
| `photorealistic` | Harmonize the object to match the photorealistic background | `photorealistic-e15.safetensors` |
| `artistic` | Stylize the object to match the artistic background | `artistic-e15.safetensors` |

The mode is fixed after construction; create a separate instance to switch styles.

**Inputs** (background image, foreground image, bbox `[x1,y1,x2,y2]`, optional foreground mask):

- Background: resized to ~1MP (aspect-preserving, LANCZOS), cropped to the 32-pixel training bucket, the bbox is mapped accordingly and drawn as a red box (with red-edge detection to avoid double drawing when the background already contains one).
- Foreground: if a mask is given (>127 is foreground), the object is cropped by the mask bounding rectangle, composited onto a white canvas, and centered into a 1024x1024 white-background image. **Without a mask, the foreground must already be a 1024x1024 white-background image.** There is no automatic segmentation.

The inference uses a fixed protocol: 20 steps, `guidance_scale=1.0`, `lora_scale=1.0`, default seed 2511. The output is an `N x H x W x 3` uint8 **BGR** array whose size follows the background aspect ratio. The background should be an **empty scene**: the model does not remove an object already present inside the bbox.

## Getting Started

```python
from libcom import QwenEditModel

net = QwenEditModel(
    device=0,
    model_type='photorealistic',
    model_path='/path/to/Qwen-Image-2.1',
    lora_path='/path/to/photorealistic-e15.safetensors',
)
results = net(background, foreground, bbox=[x1, y1, x2, y2], foreground_mask=mask)  # N,H,W,3 uint8 BGR
```

`model_path` / LoRA paths can also be given through the environment variables `LIBCOM_QWEN21_MODEL_PATH`, `LIBCOM_QWEN21_ARTISTIC_LORA_PATH` and `LIBCOM_QWEN21_PHOTOREALISTIC_LORA_PATH`.

## Model Weights

| Content | Size |
|---|---|
| Base model `Qwen-Image-2.1` (HuggingFace directory) | ~31 GB |
| `artistic-e15.safetensors` | ~161 MB |
| `photorealistic-e15.safetensors` | ~161 MB |

The LoRA adapters are not bundled with the package. If no local path is given, the default adapters are downloaded automatically through the libcom pretrained-model mechanism (`BCMIZB/Libcom_pretrained_models`). The base model is an external public checkpoint: when no local path is provided, it is fetched directly from the official HuggingFace id `Qwen/Qwen-Image-2.1` via `from_pretrained` (~31GB, first use only).

## Performance Comments

- Requires `diffusers>=0.41.0` (for `QwenImage21Pipeline`) and `transformers>=5.x`. **Note:** these versions conflict with the versions required by other libcom modules (e.g. `QwenScoreModel` pins `huggingface_hub<1.0` while this model needs `>=1.32.0`). Use a dedicated environment or plan dependency isolation accordingly.
- Verified on an A6000 (48 GB): without CPU offload a 20-step generation takes ~23.5 s with peak reserved memory of ~47 GiB. Pass `cpu_offload=True` to reduce memory usage at the cost of speed. Do not keep two full no-offload instances on the same GPU.
- The DiffSynth-format LoRA is loaded through a strict converter which validates 32 transformer blocks, 224 modules, 448 tensors and rank 32 before attaching to the pipeline.
