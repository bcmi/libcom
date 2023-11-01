import os
import yaml

def load_yaml_params(config_file):
    with open(config_file, 'r') as yaml_file:
        params = yaml.full_load(yaml_file.read())
        return params

class Config:

    def __init__(self, config_file=None):
        if config_file is None:
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            self.config_file = os.path.join(cur_dir, 'config_sfosd.yaml')
        else:
            self.config_file = config_file
        self.refresh_params()

    def refresh_params(self, config_file=None):
        if config_file is not None:
            self.config_file = config_file
        self.load_params_from_yaml()

    def print_yaml_params(self):
        params = load_yaml_params(self.config_file)
        print('*'*30, os.path.basename(self.config_file), '*'*30)
        for k, v in params.items():
            print(k, v)
        print('*'*30, os.path.basename(self.config_file), '*'*30)

    def load_params_from_yaml(self):
        # add parameters from yaml file
        names = self.__dict__
        params = load_yaml_params(self.config_file)
        for k, v in params.items():
            names[k] = v
    
    def generate_path(self):
        prefix = '{}class-bg{}-{}'.format(self.num_classes, self.encoder_feature, self.distill_type)
        if self.encoder_classify:
            prefix += '-encodercls'
        exp_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'experiments', 'student')
        os.makedirs(exp_root, exist_ok=True)
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
        # print('Experiment name {} \n'.format(os.path.basename(exp_path)))
        self.exp_name = exp_name
        self.exp_path = exp_path
        self.checkpoint_dir = os.path.join(exp_path, 'checkpoints')
        self.log_dir  = os.path.join(exp_path,  'logs')
        self.code_dir = os.path.join(exp_path, 'code')

    def create_path(self):
        # self.generate_path()
        print('Create experiment directory: ', self.exp_path)
        os.makedirs(self.exp_path, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.code_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)