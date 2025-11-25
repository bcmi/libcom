<div align="center">
</br>
<img src="https://raw.githubusercontent.com/bcmi/libcom/main/resources/LOGO.png" width="200" />

</div>

<h1 align="center">libcom: everything about image composition</h1>

</br>

[![PyPI](https://img.shields.io/pypi/v/libcom)](https://pypi.org/project/libcom)
[![Downloads](https://static.pepy.tech/badge/libcom)](https://pepy.tech/project/libcom)
[![Hits](https://hits.sh/github.com/bcmi/libcom.svg?label=views)](https://hits.sh/github.com/bcmi/libcom/)
[![Static Badge](https://img.shields.io/badge/Online_Workbench-yellow)](http://libcom.ustcnewly.com/)
[![Static Badge](https://img.shields.io/badge/Dataset&Code_Resources-green)](https://github.com/bcmi/Awesome-Object-Insertion)
[![Static Badge](https://img.shields.io/badge/survey-arxiv%3A2106.14490-red)](https://arxiv.org/pdf/2106.14490.pdf)
[![Static Badge](https://img.shields.io/github/stars/bcmi/libcom.svg?style=social)](https://github.com/bcmi/libcom/stargazers)


---

## Introduction
**_libcom_ (the library of image composition) is an image composition toolbox.** The goal of [image composition](https://github.com/bcmi/Awesome-Image-Composition) ([object insertion](https://github.com/bcmi/Awesome-Object-Insertion)) is inserting one foreground into a background image to get a realistic composite image, by addressing the inconsistencies (appearance, geometry, and semantic inconsistency) between foreground and background. Generally speaking, image composition could be used to combine the visual elements from different images.
<div align="center">
</br>
<img src="https://raw.githubusercontent.com/bcmi/libcom/main/resources/image_composition_task.gif" width="600" />
</div>

**_libcom_ covers a diversity of related tasks in the field of image composition**, including image blending, standard/painterly image harmonization, shadow generation, object placement, generative composition, quality evaluation, *etc*. For each task, we integrate one or two selected methods considering both efficiency and effectiveness. The selected methods will be continuously updated upon the emergence of better methods. 

**The ultimate goal of this library is solving all the problems related to image composition with simple `import libcom`.**

Welcome to follow WeChat public account ["Newly AIGCer"](https://www.ustcnewly.com/blog.html) or Zhihu Column ["Newly CVer"](https://www.zhihu.com/column/c_1333918224900206592) to get the latest information about image composition! 

## Online Workbench

We have built the [image composition workbench](http://libcom.ustcnewly.com/) based on our Libcom toolbox. The interface is as follows:

<div align="center">
</br>
<img src="https://raw.githubusercontent.com/bcmi/libcom/main/resources/online_workbench.jpg" width="1000" />
</div>


## Main Functions

- **get_composite_image** generates composite images using naive copy-and-paste followed by image blending.
- **OPAScoreModel** [[OPA]](https://github.com/bcmi/Object-Placement-Assessment-Dataset-OPA) evaluates the rationality of foreground object placement in a composite image.
- **FOPAHeatMapModel** [[FOPA]](https://github.com/bcmi/FOPA-Fast-Object-Placement-Assessment) can predict the rationality scores for all locations/scales given a background-foreground pair, and output the composite image with optimal location/scale.  
- **color_transfer** adjusts the foreground color to match the background using traditional color transfer method.
- **ImageHarmonizationModel** [[PCTNet]](https://github.com/rakutentech/PCT-Net-Image-Harmonization) adjusts the foreground illumination to be compatible the background given photorealistic background and photorealistic foreground.
- **PainterlyHarmonizationModel** [[PHDNet]](https://github.com/bcmi/PHDNet-Painterly-Image-Harmonization)  [[PHDiffusion]](https://github.com/bcmi/PHDiffusion-Painterly-Image-Harmonization) adjusts the foreground style to be compatible with the background given artistic background and photorealistic foreground.
- **HarmonyScoreModel** [[BargainNet]](https://github.com/bcmi/BargainNet-Image-Harmonization) evaluates the harmony level between foreground and background in a composite image.
- **InharmoniousLocalizationModel** [[MadisNet]](https://github.com/bcmi/MadisNet-Inharmonious-Region-Localization) localizes the inharmonious region in a synthetic image.
- **FOSScoreModel** [[DiscoFOS]](https://github.com/bcmi/Foreground-Object-Search-Dataset-FOSD) evaluates the compatibility between foreground and background in a composite image in terms of geometry and semantics. Due to limited training data, the generalization ability of this model is poor. 
- **ShadowGenerationModel** [[GPSDiffusion]](https://github.com/bcmi/GPSDiffusion-Object-Shadow-Generation) generates plausible shadow for the inserted object in a composite image. 
- **ReflectionGenerationModel** generates plausible reflection for the inserted object in a composite image. 
- **KontextBlendingHarmonizationModel** [[FluxKontext]](https://github.com/black-forest-labs/flux) is a generative image composition model which inserts foreground into the specified bounding box in the background. The pose and view of foreground stay unchanged. The "blending" mode does not adjust the foreground illumination, while the "harmonization" mode adjusts the foreground illumination to make the composite image harmonious. 
- **InsertAnythingModel** [[InsertAnything]](https://github.com/song-wensong/insert-anything) is a generative image composition model which inserts foreground into the specified bounding box in the background. The model has reasonable ability to adjust the pose and view of foreground according to the background. 


**For the detailed method descriptions, code examples, visualization results, and performance comments, please refer to our [[documents]](https://libcom.readthedocs.io/en/latest/).** If the model performance is not satisfactory, you can finetune the pretrained model on your own dataset using the source repository and replace the original model. 

## Requirements

The main branch is built on the Linux system with **Python 3.10** and **PyTorch>=2.6**. For other dependencies, please refer to [[requirements]](https://github.com/bcmi/libcom/blob/main/requirements.txt).

## Get Started
Please refer to [[Installation]](https://github.com/bcmi/libcom/blob/main/docs/get_started.md) for installation instructions and [[documents]](https://libcom.readthedocs.io/en/latest/) for user guidance.

## Contributors
- Institution: [Brain-like Computing and Machine Intelligence (BCMI) Lab](https://bcmi.sjtu.edu.cn/).
- Project Initiator & Team Manager: [Li Niu](https://www.ustcnewly.com/index.html). 
- Architect & Lead Developer: [Bo Zhang](https://bo-zhang-cs.github.io/), [Yujie Zhou](https://github.com/YujieOuO).
- Documentation Manager: [Jiacheng Sui](https://github.com/charlessjc).
- Module Developers: [Jiacheng Sui](https://github.com/charlessjc), [Bingjie Gao](https://github.com/WhynotHAHA), [Lingxiao Lu](https://github.com/pokaaa), [Xinhao Tao](https://github.com/taoxinhao13), [Junyan Cao](https://github.com/cjy-4), [Haonan Zhao](https://github.com/nononononno), [Jiaxuan Chen](https://github.com/csdahunzi), [Junqi You](https://github.com/dhmbb2).

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






