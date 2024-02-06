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
**Tips**: If you encounter any issues during installation related to the **`trilinear`** library, you have two choices:
1) Refer to its [official repository](https://github.com/HuiZeng/Image-Adaptive-3DLUT) and check for relevant help in the [issues section](https://github.com/HuiZeng/Image-Adaptive-3DLUT/issues).
2) If you don't need ImageHarmonizationModel, you can address the problem by blocking this function. Specifically, first comment out the [``ext_modules``](https://github.com/bcmi/libcom/blob/0987642cbbd42254c62dff988e352015510da50e/setup.py#L130) in ``setup.py``, [import code](https://github.com/bcmi/libcom/blob/0987642cbbd42254c62dff988e352015510da50e/libcom/__init__.py#L6) in ``__init__.py``, and other places that may rely on the ``trilinear`` or [``libcom/image_harmonization``](https://github.com/bcmi/libcom/tree/main/libcom/image_harmonization). Then reinstall this library by running ``python setup.py install``. 

After installation, you can verify the installation by running:
```shell
cd tests
sh run_all_tests.sh
```
The visualization results can be found in `results` folder.

## Download pretrained models
During using the toolbox, the pretrained models and related files will be automatically downloaded to the installation directory. Note downloading the pretrained models may take some time when you first call some models, especially `ShadowGenerationModel`, `ControlComModel`, and `PainterlyHarmonizationModel`.

Alternatively, you can download these files from [[Modelscope]](https://modelscope.cn/models/bcmizb/Libcom_pretrained_models/files) or [[Huggingface]](https://huggingface.co/BCMIZB/Libcom_pretrained_models/tree/main) in advance, and move them to the installation directory. The correct directory to store pretrained models can be identified from the printed message during the automatic download process. For ``ZIP`` files, don't forget to manually extract them to the installation directory. More details can be found in the [``model_download.py``](https://github.com/bcmi/libcom/blob/main/libcom/utils/model_download.py).
