"""Smoke tests for QwenEditModel preprocessing and call protocol (no real weights).

Adapted from the handoff package's test_public_api_local.py: a fake pipeline
replaces QwenImage21Pipeline so that the public __call__ path, input
preprocessing (1MP bucket + red box + 1024x1024 white foreground) and the
BGR output contract can be verified locally without model weights.
"""

import os
import sys
import unittest

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, proj_dir)

import numpy as np
from PIL import Image, ImageDraw

from libcom import QwenEditModel
from libcom.qwen_edit.qwen_edit import PROMPTS

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qwen_edit_samples')


class FakePipeline:
    _execution_device = "cpu"

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return type("Output", (), {"images": [kwargs["image"][0]]})()


class PublicApiTests(unittest.TestCase):
    def test_only_twenty_steps_are_accepted(self):
        model = QwenEditModel.__new__(QwenEditModel)
        model.pipeline = FakePipeline()
        model.model_type = "photorealistic"
        with self.assertRaisesRegex(ValueError, "20 inference steps"):
            model.infer_prepared(Image.new("RGB", (1024, 1024)),
                                 Image.new("RGB", (1024, 1024)), sample_steps=19)

    def test_raw_mask_and_background_sizes(self):
        model = QwenEditModel.__new__(QwenEditModel)
        foreground = Image.new("RGB", (120, 80), (0, 0, 255))
        mask = Image.new("L", foreground.size)
        ImageDraw.Draw(mask).rectangle((20, 10, 100, 70), fill=255)
        for width, height in ((320, 240), (2400, 1800)):
            with self.subTest(background_size=(width, height)):
                background = Image.new("RGB", (width, height), (128, 128, 128))
                bbox = (width // 4, height // 4, 3 * width // 4, 3 * height // 4)
                prepared_bg, prepared_fg = model.inputs_preprocess(
                    background, foreground, bbox, mask
                )
                self.assertLess(abs(prepared_bg.width * prepared_bg.height - 1_048_576), 100_000)
                self.assertLess(abs(prepared_bg.width / prepared_bg.height - width / height), 0.02)
                red_y, red_x = np.nonzero(np.all(np.asarray(prepared_bg) == [255, 0, 0], axis=-1))
                self.assertGreater(len(red_x), 0)
                self.assertLess(abs(red_x.min() / prepared_bg.width - 0.25), 0.03)
                self.assertLess(abs(red_y.min() / prepared_bg.height - 0.25), 0.03)
                self.assertEqual(prepared_fg.size, (1024, 1024))
                self.assertEqual(prepared_fg.getpixel((0, 0)), (255, 255, 255))
                self.assertEqual(prepared_fg.getpixel((512, 512)), (0, 0, 255))

    def test_mask_bbox_modes_and_bgr_result(self):
        model = QwenEditModel.__new__(QwenEditModel)
        model.pipeline = FakePipeline()
        model.model_type = "photorealistic"
        for sample_id in ("000000049931", "000000460450"):
            sample = os.path.join(SAMPLE_DIR, sample_id)
            bbox = [int(v) for v in open(os.path.join(sample, 'bbox.txt')).read().split()[:4]]
            for mode in ("photorealistic", "artistic"):
                with self.subTest(sample_id=sample_id, mode=mode):
                    model.model_type = mode
                    result = model(os.path.join(sample, 'background.png'),
                                   os.path.join(sample, 'foreground.png'), bbox,
                                   foreground_mask=os.path.join(sample, 'foreground_mask.png'),
                                   seed=2511)
                    call = model.pipeline.calls[-1]
                    self.assertEqual(call["prompt"], PROMPTS[mode])
                    self.assertEqual(call["num_inference_steps"], 20)
                    self.assertEqual(tuple(call["image"][1].size), (1024, 1024))
                    self.assertEqual(result.shape[0], 1)
                    self.assertEqual(result.dtype, np.uint8)
                    self.assertEqual(result.shape[1:3],
                                     (call["image"][0].height, call["image"][0].width))
                    self.assertTrue(np.any(np.all(np.asarray(call["image"][0]) == [255, 0, 0], axis=-1)))
                    self.assertEqual(call["image"][1].getpixel((0, 0)), (255, 255, 255))
                    expected_bgr = np.asarray(call["image"][0])[0, 0, ::-1]
                    np.testing.assert_array_equal(result[0, 0, 0], expected_bgr)


if __name__ == "__main__":
    unittest.main()
