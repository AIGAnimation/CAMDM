import os
import numpy as np
import random
import torch
import torch.distributed as dist

def select_platform(select_platform):
    
    # dist.init_process_group(backend="nccl")   
    if select_platform == 32:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_default_dtype(torch.float32)
        np_dtype = np.float32
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_default_dtype(torch.float64)
        np_dtype = np.float64

    return np_dtype


def to_device(data, device):
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device)
    return data.to(device)


def to_cpu(value):
    return value.detach().cpu()


def fixseed(seed):
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def array2string(array):
    output_str = ''
    for i_idx in range(array.shape[0]):
        output_str += '%06f ' % array[i_idx]
    return output_str

def mkdir(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

def save_logs(log_str, save_path):
    with open(save_path, 'w') as f:
        for line in log_str:
            f.writelines(line + '\n')

def save_errors(mpjpe_framed, save_path):
    with open(save_path, 'w') as f:
        frames = mpjpe_framed.shape[0]
        for f_idx in range(frames):
            f.writelines('%06f\n' % mpjpe_framed[f_idx]) 

def save_as_wav(frames, output_path, rate=16000):
    import scipy.io.wavfile as wavfile
    frames = frames.flatten()
    frames *= np.power(2, 15)
    frames = frames.astype(np.int16)
    wavfile.write(output_path, rate, frames)

def save_latents(latents, save_path):
    with open(save_path, 'w') as f:
        frames = latents.shape[0]
        for f_idx in range(frames):
            output_str = ''
            for i_idx in range(latents.shape[1]):
                output_str += '%08f ' % latents[f_idx][i_idx]
            f.writelines(output_str + '\n') 

def save_logs_img(log_str, save_path):
    import matplotlib.pyplot as plt
    losses = []
    for line in log_str:
        losses.append(float(line[-8:]))
    plt.plot(range(0, len(losses)), losses, 'g', label='Fitting loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)