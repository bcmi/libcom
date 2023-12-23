# GET STARTED

## Create runtime environment

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
**Tips**: We have validated the above process on Linux. You may encounter `ResolvepackageNotFound` error during installation on Windows or other systems. To resolve this, you can try removing the packages under "ResolvepackageNotFound" from libcom.yaml, then create a conda environment. Subsequently, based on the runtime error messages, use pip to install the missing packages.

## Installation
```shell
pip install libcom
```
or
```shell
python setup.py install
```
**Tips**: If you encounter any issues during installation related to the **`trilinear`** library, please refer to its [official repository](https://github.com/HuiZeng/Image-Adaptive-3DLUT) and check for relevant help in the [issues section](https://github.com/HuiZeng/Image-Adaptive-3DLUT/issues).

After installation, you can verify the installation by running:
```shell
cd tests
sh run_all_tests.sh
```
The visualization results can be found in `results` folder.

## Download pretrained models
During using the toolbox, the pretrained models and related files will be automatically downloaded to the installation directory. Note downloading the pretrained models may take some time when you first call some models, especially `ShadowGenerationModel`, `ControlComModel`, and `PainterlyHarmonizationModel`.

Alternatively, you can download these files from [[Modelscope]](https://modelscope.cn/models/bcmizb/Libcom_pretrained_models/files) or [[Huggingface]](https://huggingface.co/BCMIZB/Libcom_pretrained_models/tree/main) in advance, and move them to the installation directory.
