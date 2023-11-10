<div align="center">
</br>
<img src="resources/LOGO.png" width="200" />

</div>

<h1 align="center">libcom: everything about image composition</h1>

</br>

[![PyPI](https://img.shields.io/pypi/v/libcom)](https://pypi.org/project/libcom)
[![Downloads](https://static.pepy.tech/badge/libcom)](https://pepy.tech/project/libcom)
[![Hits](https://hits.sh/github.com/bcmi/libcom.svg?label=views&extraCount=239)](https://hits.sh/github.com/bcmi/libcom/)
![GitHub Repo stars](https://img.shields.io/github/stars/bcmi/libcom)
[![Static Badge](https://img.shields.io/badge/Image%20Composition%20Demo-Green)](https://bcmi.sjtu.edu.cn/home/niuli/demo_image_composition/)
[![Static Badge](https://img.shields.io/badge/survey-arxiv%3A2106.14490-red)](https://arxiv.org/pdf/2106.14490.pdf)
[![GitHub](https://img.shields.io/github/license/bcmi/libcom)](https://github.com/bcmi/libcom/blob/main/LICENSE)

## Introduction
**_libcom_** is an image composition toolbox. The goal of image composition is inserting one foreground into a background image to get a realistic composite image. Generally speaking, image composition could be used to combine the visual elements from different images.

**_libcom_** covers various related tasks in the field of image composition, including image harmonization, painterly image harmonization, shadow generation, object placement, generative composition, quality evaluation, *etc*. For each task, we integrate one or two selected methods considering both efficiency and effectiveness. The selected methods will be continuously updated upon the emergence of better methods. 

<details open>
<summary>Main Parts</summary>

- **get_composite_image** generates composite images using naive copy-and-paste followed by image blending.
- **OPAScoreModel** is an object placement assessment model that evaluates the rationality of object placement by predicting a rationality scores.
- **FOPAHeatMapModel** can predict the rationality scores for all locations with a pair of background and scaled foreground as input in a single forward pass.
- **color_transfer** tranfers the color of foreground to fit background scene using reinhard algorithm.
- **ImageHarmonizationModel** contains several pretrained models for image harmonization, which aims to adjust
the illumination statistics of foreground to fit background.
- **PainterlyHarmonizationModel** contains serveral pretrained models for painterly image harmonization, which aims to adjust the foreground style of the painterly composite image to make it compatible with the background.
- **HarmonyScoreModel** predicts harmony score for a composite image, in which larger harmony score implies more harmonious composite image.
- **InharmoniousLocalizationModel** supports the localization of the inharmonious region in a synthetic image.
- **FOSScoreModel** contains two foreground object search models, which can be used to evaluate the compatibility between foreground and background in terms of geometry and semantics.
- **ControlComModel** is a controllable image composition model, which unifies image blending and image harmonization in one diffusion model. 
- **ShadowGenerationModel** takes in deshadowed composite image and foreground object mask, and generates images with semantically plausible foreground shadows.
</details>

For more detailed user guidance and method description, please refer to our [[documents]](https://libcom.readthedocs.io/en/latest/). 

## Requirements

The main branch is built on the Linux system with **Python 3.8** and **PyTorch 1.10.1**. For other dependencies, please refer to [[conda_env]](requirements/libcom.yaml) and [[runtime_dependencies]](requirements/runtime.txt).

## Get Started
Please refer to [[Installation]](docs/get_started.md) for installation instructions and [[documents]](https://libcom.readthedocs.io/en/latest/) for user guides.

## Contributors
- Institution: [Brain-like Computing and Machine Intelligence (BCMI) Lab](https://bcmi.sjtu.edu.cn/).
- Project Initiator & Team Manager: [Li Niu](https://www.ustcnewly.com/index.html). 
- Architect & Lead Developer: [Bo Zhang](https://bo-zhang-cs.github.io/).   
- Documentation Manager: [Jiacheng Sui](https://github.com/charlessjc).
- Module Developers: [Jiacheng Sui](https://github.com/charlessjc), [Binjie Gao](https://github.com/WhynotHAHA), [Lingxiao Lu](https://github.com/pokaaa), [Xinhao Tao](https://github.com/taoxinhao13), [Junyan Cao](https://github.com/cjy-4).

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Bibtex

If you use our toolbox, please cite our survey paper using the following BibTeX  [[arxiv](https://arxiv.org/pdf/2106.14490.pdf)]:

```
@article{niu2021making,
  title={Making images real again: A comprehensive survey on deep image composition},
  author={Niu, Li and Cong, Wenyan and Liu, Liu and Hong, Yan and Zhang, Bo and Liang, Jing and Zhang, Liqing},
  journal={arXiv preprint arXiv:2106.14490},
  year={2021}
}
```
