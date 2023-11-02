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

For more detailed user guidance and method description, please refer to our [[documents]](https://libcom.readthedocs.io/en/latest/). 

## Usage

### Create runtime environment

```shell
git clone https://github.com/bcmi/libcom.git
cd libcom/requirements
conda env create -f libcom.yaml
conda activate Libcom
pip install -r runtime.txt # -i https://pypi.tuna.tsinghua.edu.cn/simple
# install a specific version of taming-transformers from source code
cd ../libcom/controllable_composition/source/ControlCom/src/taming-transformers
python setup.py install
```

### Installation
```shell
pip install libcom
```
or
```shell
python setup.py install
```
After that, you can verify the installation by running:
```shell
cd tests
sh run_all_tests.sh
```
The visualization results can be found in `results` folder.

### Download pretrained models
During using the toolbox, the pretrained models and related files will be automatically downloaded to the installation directory. Note downloading the pretrained models may take some time when you first call some models, especially `ControlComModel` and `PainterlyHarmonizationModel`.

Alternatively, you can download these files from [[Modelscope]](https://modelscope.cn/models/bcmizb/Libcom_pretrained_models/files) or [[Huggingface]](https://huggingface.co/BCMIZB/Libcom_pretrained_models/tree/main) in advance, and move them to the installation directory.

## Contributors

<a href="https://github.com/bcmi/libcom/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=bcmi/libcom" />
</a>

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
