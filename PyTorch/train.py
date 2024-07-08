import os
import time
import torch
import shutil
import argparse
import utils.common as common

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# import torch.distributed as dist

from utils.logger import Logger
from network.models import MotionDiffusion
from network.training import MotionTrainingPortal
from network.dataset import MotionDataset

from diffusion.create_diffusion import create_gaussian_diffusion
from config.option import add_model_args, add_train_args, add_diffusion_args, config_parse

def train(config, resume, logger, tb_writer):
    
    common.fixseed(1024)
    np_dtype = common.select_platform(32)
    
    print("Loading dataset..")
    train_data = MotionDataset(config.data, config.arch.rot_req, 
                                  config.arch.offset_frame,  config.arch.past_frame, 
                                  config.arch.future_frame, dtype=np_dtype, limited_num=config.trainer.load_num)
    train_dataloader = DataLoader(train_data, batch_size=config.trainer.batch_size, shuffle=True, num_workers=config.trainer.workers, drop_last=False, pin_memory=True)
    logger.info('\nTraining Dataset includins %d clip, with %d frame per clip;' % (len(train_data), config.arch.clip_len))
    
    diffusion = create_gaussian_diffusion(config)
    
    input_feats = (train_data.joint_num+1) * train_data.per_rot_feat   # use the root translation as an extra joint

    model = MotionDiffusion(input_feats, len(train_data.style_set),
                train_data.joint_num+1, train_data.per_rot_feat, 
                config.arch.rot_req, config.arch.clip_len,
                config.arch.latent_dim, config.arch.ff_size, 
                config.arch.num_layers, config.arch.num_heads, 
                arch=config.arch.decoder, cond_mask_prob=config.trainer.cond_mask_prob, device=config.device).to(config.device)
    
    # logger.info('\nModel structure: \n%s' % str(model))
    trainer = MotionTrainingPortal(config, model, diffusion, train_dataloader, logger, tb_writer)
    
    if resume is not None:
        try:
            trainer.load_checkpoint(resume)
        except FileNotFoundError:
            print('No checkpoint found at %s' % resume); exit()
    
    trainer.run_loop()


if __name__ == '__main__':
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='### Generative Locamotion Training')
    
    # Runtime parameters
    parser.add_argument('-n', '--name', default='debug', type=str, help='The name of this training')
    parser.add_argument('-c', '--config', default='./config/default.json', type=str, help='config file path (default: None)')
    parser.add_argument('-i', '--data', default='data/pkls/100style.pkl', type=str)
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    parser.add_argument('-s', '--save', default='./save', type=str, help='show the debug information')
    parser.add_argument('--cluster', action='store_true', help='train with GPU cluster')    
    add_model_args(parser); add_diffusion_args(parser); add_train_args(parser)
    
    args = parser.parse_args()
    
    if args.cluster:
        # If the 'cluster' argument is provided, modify the 'data' and 'save' path to match your own cluster folder location
        args.data = 'xxxxxxx/pkls/' + args.data.split('/')[-1]
        args.save = 'xxxxx'
    
    if args.config:
        config = config_parse(args)
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    if 'debug' in args.name:
        config.arch.offset_frame = config.arch.clip_len
        config.trainer.workers = 1
        config.trainer.load_num = -1
        config.trainer.batch_size = 256


    if not args.cluster:
        if os.path.exists(config.save) and 'debug' not in args.name and args.resume is None:
            allow_cover = input('Model file detected, do you want to replace it? (Y/N)')
            allow_cover = allow_cover.lower()
            if allow_cover == 'n':
                exit()
            else:
                shutil.rmtree(config.save, ignore_errors=True)
    else:
        if os.path.exists(config.save):
            if os.path.exists('%s/best.pt' % config.save):
                args.resume = '%s/best.pt' % config.save
            else:
                existing_pths = [val for val in os.listdir(config.save) if 'weights_' in val ]
                if len(existing_pths) > 0:
                    epoches = [int(filename.split('_')[1].split('.')[0]) for filename in existing_pths]
                    args.resume = '%s/%s' % (config.save, 'weights_%s.pt' % max(epoches))
    
    os.makedirs(config.save, exist_ok=True)

    logger = Logger('%s/log.txt' % config.save)
    tb_writer = SummaryWriter(log_dir='%s/runtime' % config.save)
    
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open('%s/config.json' % config.save, 'w') as f:
        f.write(str(config))
    f.close() 
    
    logger.info('\Generative locamotion training with config: \n%s' % config)
    train(config, args.resume, logger, tb_writer)
    logger.info('\nTotal training time: %s mins' % ((time.time() - start_time) / 60))
    
    
