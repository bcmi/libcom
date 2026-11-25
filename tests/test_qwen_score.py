from common import *
from libcom import QwenScoreModel
import os
import cv2
import shutil
import unittest
from unittest.mock import patch
from PIL import Image

from libcom.qwen_score import qwen_score_prediction

task_name = 'qwen_score_prediction'

# 三个选项分别对应三个 LoRA：'QwenVL-OPA' / 'QwenVL-FOS' / 'QwenVL-Harmony'
MODEL_TYPES = {
    'QwenVL-OPA':     802816,
    'QwenVL-FOS':     802816,
    'QwenVL-Harmony': 802816,
}


class QwenScoreTTATests(unittest.TestCase):
    def make_model(self, model_type, **kwargs):
        with patch.object(qwen_score_prediction, 'check_gpu_device', return_value='cpu'), \
             patch.object(qwen_score_prediction, 'resolve_base_model', return_value='base'), \
             patch.object(qwen_score_prediction, 'resolve_lora_path', return_value='lora'), \
             patch.object(QwenScoreModel, 'build_processor'), \
             patch.object(QwenScoreModel, 'build_pretrained_model'):
            return QwenScoreModel(model_type=model_type, **kwargs)

    def test_tta_is_disabled_by_default(self):
        image = Image.new('RGB', (12, 12))
        mask = Image.new('RGB', image.size)
        for model_type in MODEL_TYPES:
            with self.subTest(model_type=model_type):
                model = self.make_model(model_type)
                with patch.object(model, 'score_single', return_value=0.4) as score, \
                     patch.object(qwen_score_prediction, 'build_tta_views') as build_views:
                    self.assertEqual(model(image, mask), 0.4)
                    score.assert_called_once()
                    build_views.assert_not_called()

    def test_explicit_tta_uses_task_aggregation(self):
        image = Image.new('RGB', (12, 12))
        mask = Image.new('RGB', image.size)
        for model_type, expected in (('QwenVL-OPA', 0.5), ('QwenVL-FOS', 0.5),
                                     ('QwenVL-Harmony', 0.4)):
            with self.subTest(model_type=model_type):
                model = self.make_model(model_type, enable_tta=True)
                with patch.object(model, 'score_single', side_effect=[0.4, 0.9, 0.2]) as score:
                    self.assertAlmostEqual(model(image, mask), expected)
                    self.assertEqual(score.call_count, 3)


if __name__ == '__main__':
    test_set = get_test_list()
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', task_name)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)

    for model_type, max_pixels in MODEL_TYPES.items():
        print(f'begin testing {task_name} | {model_type} ...')
        net = QwenScoreModel(device=0, model_type=model_type, max_pixels=max_pixels)
        for pair in test_set:
            comp_img, comp_mask = pair['composite'], pair['composite_mask']
            score = net(comp_img, comp_mask)
            img_name = os.path.basename(comp_img).replace('.png', '.jpg')
            grid_img = make_image_grid([comp_img, comp_mask],
                                       text_list=[f'{model_type}:{score:.2f}', 'composite-mask'])
            res_path = os.path.join(result_dir, f'{model_type}_{img_name}')
            cv2.imwrite(res_path, grid_img)
            print('save result to ', res_path)
        print(f'end testing {task_name} | {model_type}!\n')
