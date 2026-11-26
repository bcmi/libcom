# -*- coding: utf-8 -*-
"""
demo.py — ReflectionGenerationModel 最小运行示例
================================================

给定「合成图 + 前景物体掩码」，生成带倒影的图并保存到本地。

用法
----
    # 用内置样例（tests/reflection_generation_source）跑一张，结果存到 results/reflection_demo.png
    python demo.py

    # 指定自己的输入与输出
    python demo.py \
        --composite /path/to/composite.png \
        --mask      /path/to/foreground_mask.png \
        --output    results/my_result.png \
        --device    0

参数
----
    --composite  合成图路径（有物体、无倒影的原图）
    --mask       前景物体掩码路径（单通道二值图，物体为白色）
    --output     输出图片路径（默认 results/reflection_demo.png）
    --device     使用的 GPU 编号（默认 0）
    --number     生成张数（默认 1；>1 时会保存 <名>_1.png ... 并返回多张）

输出
----
    512x512 的 BGR PNG 图片，倒影区域由「后处理网络 + 泊松融合」无缝贴回原图。
"""

import os
import argparse

import cv2

from libcom import ReflectionGenerationModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description='ReflectionGeneration demo')
    parser.add_argument('--composite', type=str,
                        default=os.path.join(SCRIPT_DIR, 'tests', 'reflection_generation_source', 'composite', '1.png'),
                        help='合成图路径（有物体、无倒影）')
    parser.add_argument('--mask', type=str,
                        default=os.path.join(SCRIPT_DIR, 'tests', 'reflection_generation_source', 'composite_mask', '1.png'),
                        help='前景物体掩码路径（物体为白色）')
    parser.add_argument('--output', type=str,
                        default=os.path.join(SCRIPT_DIR, 'results', 'reflection_demo.png'),
                        help='输出图片路径')
    parser.add_argument('--device', type=int, default=0, help='GPU 编号，默认 0')
    parser.add_argument('--number', type=int, default=1, help='生成张数，默认 1')
    return parser.parse_args()


def main():
    args = parse_args()

    for path in (args.composite, args.mask):
        if not os.path.exists(path):
            raise FileNotFoundError(f'输入不存在: {path}')

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    # 只需指定 model_type；权重默认从 reflection_generation/pretrained_models/ 读取
    net = ReflectionGenerationModel(device=args.device, model_type='ReflectionGeneration')
    preds = net(args.composite, args.mask, number=args.number)

    if len(preds) == 1:
        cv2.imwrite(args.output, preds[0])
        print(f'[done] 已保存: {os.path.abspath(args.output)}')
    else:
        stem = os.path.splitext(args.output)[0]
        for i, pred in enumerate(preds, 1):
            path = f'{stem}_{i}.png'
            cv2.imwrite(path, pred)
            print(f'[done] 已保存: {os.path.abspath(path)}')


if __name__ == '__main__':
    main()
