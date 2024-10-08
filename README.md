<div align="center">
</br>
<img src="https://raw.githubusercontent.com/bcmi/libcom/main/resources/LOGO.png" width="200" />

</div>

<h1 align="center">libcom: everything about image composition</h1>

</br>

[![PyPI](https://img.shields.io/pypi/v/libcom)](https://pypi.org/project/libcom)
[![Downloads](https://static.pepy.tech/badge/libcom)](https://pepy.tech/project/libcom)
[![Hits](https://hits.sh/github.com/bcmi/libcom.svg?label=views)](https://hits.sh/github.com/bcmi/libcom/)
[![Static Badge](https://img.shields.io/badge/Image%20Composition%20Demo-Green)](https://bcmi.sjtu.edu.cn/home/niuli/demo_image_composition/)
[![Static Badge](https://img.shields.io/badge/Resources-Awesome-green?style=flat&label=Resources)](https://github.com/bcmi/Awesome-Image-Composition)
[![Static Badge](https://img.shields.io/badge/survey-arxiv%3A2106.14490-red)](https://arxiv.org/pdf/2106.14490.pdf)

**We co-founded a startup company [miguo.ai](https://miguocomics.com/), dedicated to accelerating the production of comics and animations using AIGC technology. If you are seeking internship or full-time positions, please feel free to send your resume to hr@miguocomics.com.**

---

## Introduction
**_libcom_ (the library of image composition) is an image composition toolbox.** The goal of image composition (object insertion) is inserting one foreground into a background image to get a realistic composite image, by addressing the inconsistencies (appearance, geometry, and semantic inconsistency) between foreground and background. Generally speaking, image composition could be used to combine the visual elements from different images.
<div align="center">
</br>
<img src="https://raw.githubusercontent.com/bcmi/libcom/main/resources/image_composition_task.gif" width="600" />
</div>

**_libcom_ covers a diversity of related tasks in the field of image composition**, including image blending, standard/painterly image harmonization, shadow generation, object placement, generative composition, quality evaluation, *etc*. For each task, we integrate one or two selected methods considering both efficiency and effectiveness. The selected methods will be continuously updated upon the emergence of better methods. 

**The ultimate goal of this library is solving all the problems related to image composition with simple `import libcom`.**

### Main Functions

- **get_composite_image** generates composite images using naive copy-and-paste followed by image blending.
- **OPAScoreModel** [[OPA]](https://github.com/bcmi/Object-Placement-Assessment-Dataset-OPA) evaluates the rationality of foreground object placement in a composite image.
- **FOPAHeatMapModel** [[FOPA]](https://github.com/bcmi/FOPA-Fast-Object-Placement-Assessment) can predict the rationality scores for all locations/scales given a background-foreground pair, and output the composite image with optimal location/scale.  
- **color_transfer** adjusts the foreground color to match the background using traditional color transfer method.
- **ImageHarmonizationModel** [[CDTNet]](https://github.com/bcmi/CDTNet-High-Resolution-Image-Harmonization)  [[PCTNet]](https://github.com/rakutentech/PCT-Net-Image-Harmonization) adjusts the foreground illumination to be compatible the background given photorealistic background and photorealistic foreground.
- **PainterlyHarmonizationModel** [[PHDNet]](https://github.com/bcmi/PHDNet-Painterly-Image-Harmonization)  [[PHDiffusion]](https://github.com/bcmi/PHDiffusion-Painterly-Image-Harmonization) adjusts the foreground style to be compatible with the background given artistic background and photorealistic foreground.
- **HarmonyScoreModel** [[BargainNet]](https://github.com/bcmi/BargainNet-Image-Harmonization) evaluates the harmony level between foreground and background in a composite image.
- **InharmoniousLocalizationModel** [[MadisNet]](https://github.com/bcmi/MadisNet-Inharmonious-Region-Localization) localizes the inharmonious region in a synthetic image.
- **FOSScoreModel** [[DiscoFOS]](https://github.com/bcmi/Foreground-Object-Search-Dataset-FOSD) evaluates the compatibility between foreground and background in a composite image in terms of geometry and semantics.
- **ShadowGenerationModel** [[GPSDiffusion]](https://github.com/bcmi/GPSDiffusion-Object-Shadow-Generation) generates plausible shadow for the inserted object in a composite image. This model is unstable and you can pick the most satisfactory one from multiple generated results. 
- **ControlComModel** [[ControlCom]](https://github.com/bcmi/ControlCom-Image-Composition) is a generative image composition model which unifies image blending and image harmonization. The pose and view of foreground stay unchanged. 
- **MureObjectStitchModel** [[MureObjectStitch]](https://github.com/bcmi/MureObjectStitch-Image-Composition) is another generative image composition model which can adjust the pose and view of foreground. It supports multiple reference images of one foreground object. If you have a few images containing the foreground object, we suggest finetuning MureObjectStitch using these images for better detail preservation. 


**For the detailed method descriptions, code examples, visualization results, and performance comments, please refer to our [[documents]](https://libcom.readthedocs.io/en/latest/).** If the model performance is not satisfactory, you can finetune the pretrained model on your own dataset using the source repository and replace the original model. 

## Requirements

The main branch is built on the Linux system with **Python 3.8** and **PyTorch>=1.10.1**. For other dependencies, please refer to [[conda_env]](https://github.com/bcmi/libcom/blob/main/requirements/libcom.yaml) and [[runtime_dependencies]](https://github.com/bcmi/libcom/blob/main/requirements/runtime.txt).

## Get Started
Please refer to [[Installation]](https://github.com/bcmi/libcom/blob/main/docs/get_started.md) for installation instructions and [[documents]](https://libcom.readthedocs.io/en/latest/) for user guidance.

## Contributors
- Institution: [Brain-like Computing and Machine Intelligence (BCMI) Lab](https://bcmi.sjtu.edu.cn/).
- Project Initiator & Team Manager: [Li Niu](https://www.ustcnewly.com/index.html). 
- Architect & Lead Developer: [Bo Zhang](https://bo-zhang-cs.github.io/), [Yujie Zhou](https://github.com/YujieOuO).
- Documentation Manager: [Jiacheng Sui](https://github.com/charlessjc).
- Module Developers: [Jiacheng Sui](https://github.com/charlessjc), [Binjie Gao](https://github.com/WhynotHAHA), [Lingxiao Lu](https://github.com/pokaaa), [Xinhao Tao](https://github.com/taoxinhao13), [Junyan Cao](https://github.com/cjy-4), [Haonan Zhao](https://github.com/nononononno), [Jiaxuan Chen](https://github.com/csdahunzi), [Junqi You](https://github.com/dhmbb2).

## License

This project is released under the [Apache 2.0 license](https://github.com/bcmi/libcom/blob/main/LICENSE).

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
