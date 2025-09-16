# GET STARTED

## Create runtime environment

```shell
git clone https://github.com/bcmi/libcom.git
cd libcom
conda create -n libcom python=3.10
conda activate libcom
pip install -r requirements.txt # -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Installation
```shell
pip install libcom
or
python setup.py install
```

After installation, you can verify the installation by running:
```shell
cd tests
sh run_all_tests.sh
```
The visualization results can be found in `results` folder.

## Download pretrained models
During using the toolbox, the pretrained models and related files will be automatically downloaded to the installation directory. Note downloading the pretrained models may take some time when you first call some models.

Alternatively, you can download these files from [[Modelscope]](https://modelscope.cn/models/bcmizb/Libcom_pretrained_models/files) or [[Huggingface]](https://huggingface.co/BCMIZB/Libcom_pretrained_models/tree/main) in advance, and move them to the installation directory. The correct directory to store pretrained models can be identified from the printed message during the automatic download process. For ``ZIP`` files, don't forget to manually extract them to the installation directory. More details can be found in the [``model_download.py``](https://github.com/bcmi/libcom/blob/main/libcom/utils/model_download.py).
