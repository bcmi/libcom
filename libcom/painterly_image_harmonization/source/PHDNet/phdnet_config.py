import os

class Config(object):
    # config
    backbone = 'resnet18'
    model = 'phdnet'
    dataset_mode = 'phd'
    preprocess = 'none'
    batch_size = 1
    num_threads = 6
    gpu_ids = [2]
    load_size = 512


opt = Config()

if __name__ == "__main__":
    print(opt.exp_root)
