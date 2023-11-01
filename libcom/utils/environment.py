import torch

def check_gpu_device(device):
    assert torch.cuda.is_available(), 'Only GPU are supported'
    if isinstance(device, (int, str)):
        device = int(device)
        assert 0 <= device < torch.cuda.device_count(), f'invalid device id: {device}'
        device = torch.device(f'cuda:{device}')
    if isinstance(device, torch.device):
        return device
    else:
        raise Exception('invalid device type: type({})={}'.format(device, type(device)))