# GET STARTED

## Create runtime environment

```shell
conda create -n libcom python=3.10
conda activate libcom
```

## Installation

For most users, install from PyPI directly:

```shell
pip install libcom
```

For developers who want to modify the source code:

```shell
git clone https://github.com/bcmi/libcom.git
cd libcom
pip install -r requirements.txt
pip install -e .
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

## Notes on installation

If the network is restricted, you can use mirror installation sources.
```shell
# pip install
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# huggingface download
export HF_ENDPOINT="https://hf‑mirror.com"
```

For certain HuggingFace models (e.g., FLUX-related models), downloading requires a token, otherwise it will fail with an error.
```shell
# hugginface permission
export HF_TOKEN="hf_yourtoken"
```



