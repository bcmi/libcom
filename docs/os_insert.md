# OS Insert

`OSInsertModel` is an object insertion model that inserts a foreground object into a background image under a given bounding box.

It supports two inference modes:

- **conservative**: builds bbox mask directly on background, then runs insertion; faster and more stable.
- **aggressive**: uses ObjectStitch + SAM + insertion refinement; better flexibility for hard cases.

## API

```python
from libcom import OSInsertModel

model = OSInsertModel(device='cuda:0')
result = model(
    background_path='path/to/background.png',
    foreground_path='path/to/foreground.png',
    foreground_mask_path='path/to/foreground_mask.png',
    bbox=[1000, 895, 1480, 1355],
    result_dir='path/to/save_dir',
    mode='aggressive',  # 'conservative' or 'aggressive'
    verbose=True,
)
```

## Key Arguments

- `background_path`: background image path.
- `foreground_path`: foreground image path.
- `foreground_mask_path`: foreground mask path.
- `bbox`: insertion box in `[x1, y1, x2, y2]` format.
- `result_dir`: output directory.
- `mode`: `'conservative'` or `'aggressive'`.
- `seed`: random seed (default `123`).
- `strength`: insertion strength (default `1.0`).
- `split_ratio`: denoising split ratio (default `0.5`).
- `verbose`: whether to save intermediates into `result_dir/intermediates`.

## Output

Returns composed image as `numpy.ndarray` (or `None` when inference fails).
