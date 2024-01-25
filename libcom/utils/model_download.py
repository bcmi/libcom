import os
import shutil
from tqdm import tqdm
import zipfile

hf_repo = 'BCMIZB/Libcom_pretrained_models'
ms_repo = 'bcmizb/Libcom_pretrained_models'

def download_pretrained_model(weight_path):
    if os.path.exists(weight_path):
        assert os.path.isfile(weight_path), weight_path
        return weight_path
    else:
        weight_path= os.path.abspath(weight_path)
        model_name = os.path.basename(weight_path)
        save_dir   = os.path.dirname(weight_path)
        download_file_from_network(model_name, save_dir)
        print('Pretrained model has been stored to ', weight_path)
        return weight_path
    
def download_entire_folder(folder_path):
    if os.path.exists(folder_path):
        assert os.path.isdir(folder_path), folder_path
        assert len(os.listdir(folder_path)) > 1, f'{folder_path} is an empty folder'
        return folder_path
    else:
        folder_path = os.path.abspath(folder_path) 
        folder_name = os.path.basename(folder_path)
        file_name   = folder_name + '.zip'
        save_dir    = os.path.dirname(folder_path)
        download_file_from_network(file_name, save_dir)
        zip_file    = zipfile.ZipFile(os.path.join(save_dir, file_name))
        zip_file.extractall(save_dir)
        os.remove(os.path.join(save_dir, file_name))
        print('Folder has been stored to ', folder_path)
        return folder_path
    
def download_file_from_network(file_name, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    print('Try to download {} to {}'.format(file_name, save_dir))
    try:
        from huggingface_hub import hf_hub_download
        file_path = hf_hub_download(repo_id=hf_repo, 
                                    filename=file_name, 
                                    cache_dir=save_dir)
    except:
        from modelscope.hub.file_download import model_file_download
        file_path = model_file_download(model_id=ms_repo, 
                                        file_path=file_name, 
                                        cache_dir=save_dir, 
                                        revision='master')
    assert os.path.exists(file_path), 'Download {} failed, please try again'.format(file)
    save_path = os.path.abspath(os.path.join(save_dir, file_name))
    shutil.copyfile(os.path.abspath(file_path), save_path, follow_symlinks=True)
    assert os.path.exists(save_path), 'Move file to {} failed, please try again'.format(save_path)
    os.remove(os.path.realpath(file_path)) # delete the cache
        
if __name__ == '__main__':
    file_list   = ['BargainNet.pth', 'SOPA.pth']
    folder_list = ['openai-clip-vit-large-patch14'] 
    for file in file_list:
        weight_path = './pretrained_models/' + file
        download_pretrained_model(weight_path)
    
    for folder in folder_list:
        folder_path = './pretrained_models/' + folder
        download_entire_folder(folder_path)