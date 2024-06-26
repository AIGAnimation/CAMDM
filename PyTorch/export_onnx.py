import os
import json
import torch
import argparse
import numpy as np

import utils.common as common

from network.models import MotionDiffusion
from network.dataset import MotionDataset
from config.option import add_model_args, add_diffusion_args
from diffusion.resample import create_named_schedule_sampler
from diffusion.create_diffusion import create_gaussian_diffusion

import onnxruntime as ort
import onnx

if __name__ == "__main__":
    common.fixseed(1024)
    
    parser = argparse.ArgumentParser(description='### Generative Locamotion Exporting')
    parser.add_argument('-r', '--checkpoints', default='save/camdm_100style/best.pt', type=str, help='path to latest checkpoint (default: None)')
    add_model_args(parser); add_diffusion_args(parser); 
    args = parser.parse_args()
    
    model_name = args.checkpoints.split('/')[-2]
    check_point_name = args.checkpoints.split('/')[-1].split('.')[0]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    
    if os.path.exists(args.checkpoints):
        checkpoint = torch.load(args.checkpoints, map_location=device)
    else:
        raise AssertionError("Model file not found.")
    
    config = checkpoint['config']
    
    export_folder = os.path.join(os.path.dirname(args.checkpoints), 'onnx_%s_%s' % (model_name, check_point_name))
    common.mkdir(export_folder)
    
    config.data = 'data/pkls/' + config.data.split('pkls/')[-1]
    
    train_data = MotionDataset(config.data, config.arch.rot_req, 
                                  config.arch.offset_frame,  config.arch.past_frame, 
                                  config.arch.future_frame, dtype=np.float32, limited_num=-1)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=1, drop_last=False)
    
    input_feats = (train_data.joint_num+1) * train_data.per_rot_feat
    
    diffusion = create_gaussian_diffusion(config)
    schedule_sampler_type = 'uniform'
    schedule_sampler = create_named_schedule_sampler(schedule_sampler_type, diffusion)
    
    model = MotionDiffusion(input_feats,  len(train_data.style_set),
                train_data.joint_num+1, train_data.per_rot_feat, 
                config.arch.rot_req, config.arch.clip_len,
                config.arch.latent_dim, config.arch.ff_size, 
                config.arch.num_layers, config.arch.num_heads, 
                arch=config.arch.decoder, cond_mask_prob=config.trainer.cond_mask_prob, device=config.device).to(config.device)
  
    model.load_state_dict(checkpoint['state_dict'])
    
    onnx_path = os.path.join(export_folder, '%s_%s.onnx' % (model_name, check_point_name))
    json_path = os.path.join(export_folder, '%s_%s.config.json' % (model_name, check_point_name))
    latent_path = os.path.join(export_folder, '100style_conditions.json')
    
    datas = train_dataloader.__iter__().__next__()
    datas = {key: val.to(device) if torch.is_tensor(val) else val for key, val in datas.items()}
    cond = {key: val.to(device) if torch.is_tensor(val) else val for key, val in datas['conditions'].items()}
    x_start = datas['data'].permute(0, 2, 3, 1)
    cond['past_motion'] = cond['past_motion'].permute(0, 2, 3, 1) # [bs, joint_num, joint_feat, past_frames]
    cond['traj_pose'] = cond['traj_pose'].permute(0, 2, 1) # [bs, 6, frame_num//2]
    cond['traj_trans'] = cond['traj_trans'].permute(0, 2, 1) # [bs, 2, frame_num//2]
    
    # define input and output names for onnx model
    input_tuple = (x_start, torch.tensor([0]*1), cond['past_motion'], cond['traj_pose'], cond['traj_trans'], cond['style_idx'])
    input_names = ['input_x', 'time_steps', 'past_motion', 'traj_pose', 'traj_trans', 'style_idx'] 
    output_names = ['output']
    
    results = model(*input_tuple)
    
    # export model config
    model_config_json = {
        "past_points": config.arch.past_frame,
        "future_points": config.arch.future_frame,
        "joint_num": train_data.joint_num,
        "diffusion_steps": config.diff.diffusion_steps,
        "posterior_log_variance_clipped": diffusion.posterior_log_variance_clipped.tolist(),
        "posterior_mean_coef1": diffusion.posterior_mean_coef1.tolist(),
        "posterior_mean_coef2": diffusion.posterior_mean_coef2.tolist(),
        "joint_names": train_data.T_pose.names,
    }
    
    all_styles = train_data.style_set
    
    latents_config_json = {
        "styles": all_styles,
    }
    
    with open(json_path, 'w') as f:
        json.dump(model_config_json, f, indent=4)
        
    with open(latent_path, 'w') as f:
        json.dump(latents_config_json, f, indent=4)
    
    torch.onnx.export(model, input_tuple, onnx_path, input_names=input_names, output_names=output_names, opset_version=15, export_params=True, verbose=True)
    
    