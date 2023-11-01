import os

class Config(object):
    # path
    dataset_path = './dataset'
    
    train_data_path = os.path.join(dataset_path, 'train_set.csv')
    test_data_path = os.path.join(dataset_path, 'test_set.csv')

    img_path = os.path.join('./')
    mask_path = os.path.join('.')
 
    pretrained_model_path = './pretrained_models'

    ## 
    class_num = 2 
    img_size = 256
    batch_size = 128
    num_workers = 8
    global_feature_size = 8

    # train
    base_lr = 1e-4
    lr_milestones = [10, 16]
    lr_gamma = 0.1
    epochs = 25
    eval_freq = 1
    save_freq = 5
    display_freq = 20

    # Config
    backbone = 'resnet18'
    gpu_id = 1

    without_mask = False

    # Save path 
    prefix = backbone
    if without_mask:
        prefix += '+without_mask'

    exp_root = os.path.join(os.getcwd(), './experiments/simopa/')
    exp_name = prefix
    exp_path = os.path.join(exp_root, prefix)
    while os.path.exists(exp_path):
        index = os.path.basename(exp_path).split(prefix)[-1].split('repeat')[-1]
        try:
            index = int(index) + 1
        except:
            index = 1
        exp_name = prefix + ('_repeat{}'.format(index))
        exp_path = os.path.join(exp_root, exp_name)

    checkpoint_dir = os.path.join(exp_path, 'checkpoints')
    log_dir = os.path.join(exp_path, 'logs')

    def create_path(self):
        print('Create experiments directory: ', self.exp_path)
        os.makedirs(self.exp_path)
        os.makedirs(self.checkpoint_dir)
        os.makedirs(self.log_dir)

opt = Config()

if __name__ == "__main__":
    print(opt.exp_root)
