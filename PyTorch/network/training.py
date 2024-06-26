import numpy as np
import blobfile as bf
import utils.common as common
from tqdm import tqdm
import utils.nn_transforms as nn_transforms
import itertools

import torch
from torch.optim import AdamW
from torch.utils.data import Subset, DataLoader
from torch_ema import ExponentialMovingAverage

from diffusion.resample import create_named_schedule_sampler
from diffusion.gaussian_diffusion import *


class BaseTrainingPortal:
    def __init__(self, config, model, diffusion, dataloader, logger, tb_writer, prior_loader=None):
        
        self.model = model
        self.diffusion = diffusion
        self.dataloader = dataloader
        self.logger = logger
        self.tb_writer = tb_writer
        self.config = config
        self.batch_size = config.trainer.batch_size
        self.lr = config.trainer.lr
        self.lr_anneal_steps = config.trainer.lr_anneal_steps

        self.epoch = 0
        self.num_epochs = config.trainer.epoch
        self.save_freq = config.trainer.save_freq
        self.best_loss = 1e10
        
        print('Train with %d epoches, %d batches by %d batch_size' % (self.num_epochs, len(self.dataloader), self.batch_size))

        self.save_dir = config.save

        self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=config.trainer.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.num_epochs, eta_min=self.lr * 0.1)
        
        if config.trainer.ema:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.995)
        
        self.device = config.device

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.use_ddp = False
        
        self.prior_loader = prior_loader
        
        
    def diffuse(self, x_start, t, cond, noise=None, return_loss=False):
        raise NotImplementedError('diffuse function must be implemented')

    def evaluate_sampling(self, dataloader, save_folder_name):
        raise NotImplementedError('evaluate_sampling function must be implemented')
    
        
    def run_loop(self):
        sampling_num = 16
        sampling_idx = np.random.randint(0, len(self.dataloader.dataset), sampling_num)
        sampling_subset = DataLoader(Subset(self.dataloader.dataset, sampling_idx), batch_size=sampling_num)
        self.evaluate_sampling(sampling_subset, save_folder_name='init_samples')
        
        epoch_process_bar = tqdm(range(self.epoch, self.num_epochs), desc=f'Epoch {self.epoch}')
        for epoch_idx in epoch_process_bar:
            self.model.train()
            self.model.training = True
            self.epoch = epoch_idx
            epoch_losses = {}
            
            data_len = len(self.dataloader)
            
            for datas in self.dataloader:
                datas = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas.items()}
                cond = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas['conditions'].items()}
                x_start = datas['data']

                self.opt.zero_grad()
                t, weights = self.schedule_sampler.sample(x_start.shape[0], self.device)
                
                _, losses = self.diffuse(x_start, t, cond, noise=None, return_loss=True)
                total_loss = (losses["loss"] * weights).mean()
                total_loss.backward()
                self.opt.step()
            
                if self.config.trainer.ema:
                    self.ema.update()
                
                for key_name in losses.keys():
                    if 'loss' in key_name:
                        if key_name not in epoch_losses.keys():
                            epoch_losses[key_name] = []
                        epoch_losses[key_name].append(losses[key_name].mean().item())
            
            if self.prior_loader is not None:
                for prior_datas in itertools.islice(self.prior_loader, data_len):
                    prior_datas = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in prior_datas.items()}
                    prior_cond = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in prior_datas['conditions'].items()}
                    prior_x_start = prior_datas['data']
                    
                    self.opt.zero_grad()
                    t, weights = self.schedule_sampler.sample(prior_x_start.shape[0], self.device)
                    
                    _, prior_losses = self.diffuse(prior_x_start, t, prior_cond, noise=None, return_loss=True)
                    total_loss = (prior_losses["loss"] * weights).mean()
                    total_loss.backward()
                    self.opt.step()
                    
                    for key_name in prior_losses.keys():
                        if 'loss' in key_name:
                            if key_name not in epoch_losses.keys():
                                epoch_losses[key_name] = []
                            epoch_losses[key_name].append(prior_losses[key_name].mean().item())
            
            loss_str = ''
            for key in epoch_losses.keys():
                loss_str += f'{key}: {np.mean(epoch_losses[key]):.6f}, '
            
            epoch_avg_loss = np.mean(epoch_losses['loss'])
            
            if self.epoch > 10 and epoch_avg_loss < self.best_loss:                
                self.save_checkpoint(filename='best')
            
            if epoch_avg_loss < self.best_loss:
                self.best_loss = epoch_avg_loss
            
            epoch_process_bar.set_description(f'Epoch {epoch_idx}/{self.config.trainer.epoch} | loss: {epoch_avg_loss:.6f} | best_loss: {self.best_loss:.6f}')
            self.logger.info(f'Epoch {epoch_idx}/{self.config.trainer.epoch} | {loss_str} | best_loss: {self.best_loss:.6f}')
                        
            if epoch_idx > 0 and epoch_idx % self.config.trainer.save_freq == 0:
                self.save_checkpoint(filename=f'weights_{epoch_idx}')
                self.evaluate_sampling(sampling_subset, save_folder_name='train_samples')
            
            for key_name in epoch_losses.keys():
                if 'loss' in key_name:
                    self.tb_writer.add_scalar(f'train/{key_name}', np.mean(epoch_losses[key_name]), epoch_idx)

            self.scheduler.step()
        
        best_path = '%s/best.pt' % (self.config.save)
        self.load_checkpoint(best_path)
        self.evaluate_sampling(sampling_subset, save_folder_name='best')


    def state_dict(self):
        model_state = self.model.state_dict()
        opt_state = self.opt.state_dict()
            
        return {
            'epoch': self.epoch,
            'state_dict': model_state,
            'opt_state_dict': opt_state,
            'config': self.config,
            'loss': self.best_loss,
        }

    def save_checkpoint(self, filename='weights'):
        save_path = '%s/%s.pt' % (self.config.save, filename)
        with bf.BlobFile(bf.join(save_path), "wb") as f:
            torch.save(self.state_dict(), f)
        self.logger.info(f'Saved checkpoint: {save_path}')


    def load_checkpoint(self, resume_checkpoint, load_hyper=True):
        if bf.exists(resume_checkpoint):
            checkpoint = torch.load(resume_checkpoint)
            self.model.load_state_dict(checkpoint['state_dict'])
            if load_hyper:
                self.epoch = checkpoint['epoch'] + 1
                self.best_loss = checkpoint['loss']
                self.opt.load_state_dict(checkpoint['opt_state_dict'])
            self.logger.info('\nLoad checkpoint from %s, start at epoch %d, loss: %.4f' % (resume_checkpoint, self.epoch, checkpoint['loss']))
        else:
            raise FileNotFoundError(f'No checkpoint found at {resume_checkpoint}')


class MotionTrainingPortal(BaseTrainingPortal):
    def __init__(self, config, model, diffusion, dataloader, logger, tb_writer, finetune_loader=None):
        super().__init__(config, model, diffusion, dataloader, logger, tb_writer, finetune_loader)
        self.skel_offset = torch.from_numpy(self.dataloader.dataset.T_pose.offsets).to(self.device)
        self.skel_parents = self.dataloader.dataset.T_pose.parents
        

    def diffuse(self, x_start, t, cond, noise=None, return_loss=False):
        batch_size, frame_num, joint_num, joint_feat = x_start.shape
        x_start = x_start.permute(0, 2, 3, 1)
        
        if noise is None:
            noise = th.randn_like(x_start)
        
        x_t = self.diffusion.q_sample(x_start, t, noise=noise)
        
        # [bs, joint_num, joint_feat, future_frames]
        cond['past_motion'] = cond['past_motion'].permute(0, 2, 3, 1) # [bs, joint_num, joint_feat, past_frames]
        cond['traj_pose'] = cond['traj_pose'].permute(0, 2, 1) # [bs, 6, frame_num//2]
        cond['traj_trans'] = cond['traj_trans'].permute(0, 2, 1) # [bs, 2, frame_num//2]
        
        model_output = self.model.interface(x_t, self.diffusion._scale_timesteps(t), cond)
        
        if return_loss:
            loss_terms = {}
            
            if self.diffusion.model_var_type in [ModelVarType.LEARNED,  ModelVarType.LEARNED_RANGE]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = torch.split(model_output, C, dim=1)
                frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
                loss_terms["vb"] = self.diffusion._vb_terms_bpd(model=lambda *args, r=frozen_out: r, x_start=x_start, x_t=x_t, t=t, clip_denoised=False)["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    loss_terms["vb"] *= self.diffusion.num_timesteps / 1000.0
            target = {
                ModelMeanType.PREVIOUS_X: self.diffusion.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.diffusion.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            mask = cond['mask'].view(batch_size, 1, 1, -1)
            
            if self.config.trainer.use_loss_mse:
                loss_terms['loss_data'] = self.diffusion.masked_l2(target, model_output, mask) # mean_flat(rot_mse)
                
            if self.config.trainer.use_loss_vel:
                model_output_vel = model_output[..., 1:] - model_output[..., :-1]
                target_vel = target[..., 1:] - target[..., :-1]
                loss_terms['loss_data_vel'] = self.diffusion.masked_l2(target_vel[:, :-1], model_output_vel[:, :-1], mask[..., 1:])
                  
            if self.config.trainer.use_loss_3d or self.config.use_loss_contact:
                target_rot, pred_rot, past_rot = target.permute(0, 3, 1, 2), model_output.permute(0, 3, 1, 2), cond['past_motion'].permute(0, 3, 1, 2)
                target_root_pos, pred_root_pos, past_root_pos = target_rot[:, :, -1, :3], pred_rot[:, :, -1, :3], past_rot[:, :, -1, :3]
                skeletons = self.skel_offset.unsqueeze(0).expand(batch_size, -1, -1)
                parents = self.skel_parents[None]
                
                target_xyz = neural_FK(target_rot[:, :, :-1], skeletons, target_root_pos, parents, rotation_type=self.config.arch.rot_req)
                pred_xyz = neural_FK(pred_rot[:, :, :-1], skeletons, pred_root_pos, parents, rotation_type=self.config.arch.rot_req)
                
                if self.config.trainer.use_loss_3d:
                    loss_terms["loss_geo_xyz"] = self.diffusion.masked_l2(target_xyz.permute(0, 2, 3, 1), pred_xyz.permute(0, 2, 3, 1), mask)
                
                if self.config.trainer.use_loss_vel and self.config.trainer.use_loss_3d:
                    target_xyz_vel = target_xyz[:, 1:] - target_xyz[:, :-1]
                    pred_xyz_vel = pred_xyz[:, 1:] - pred_xyz[:, :-1]
                    loss_terms["loss_geo_xyz_vel"] = self.diffusion.masked_l2(target_xyz_vel.permute(0, 2, 3, 1), pred_xyz_vel.permute(0, 2, 3, 1), mask[..., 1:])
                
                if self.config.trainer.use_loss_contact:
                    l_foot_idx, r_foot_idx = 24, 19
                    relevant_joints = [l_foot_idx, r_foot_idx]
                    target_xyz_reshape = target_xyz.permute(0, 2, 3, 1)  
                    pred_xyz_reshape = pred_xyz.permute(0, 2, 3, 1)
                    gt_joint_xyz = target_xyz_reshape[:, relevant_joints, :, :]  # [BatchSize, 2, 3, Frames]
                    gt_joint_vel = torch.linalg.norm(gt_joint_xyz[:, :, :, 1:] - gt_joint_xyz[:, :, :, :-1], axis=2)  # [BatchSize, 4, Frames]
                    fc_mask = torch.unsqueeze((gt_joint_vel <= 0.01), dim=2).repeat(1, 1, 3, 1)
                    pred_joint_xyz = pred_xyz_reshape[:, relevant_joints, :, :]  # [BatchSize, 2, 3, Frames]
                    pred_vel = pred_joint_xyz[:, :, :, 1:] - pred_joint_xyz[:, :, :, :-1]
                    pred_vel[~fc_mask] = 0
                    loss_terms["loss_foot_contact"] = self.diffusion.masked_l2(pred_vel,
                                                torch.zeros(pred_vel.shape, device=pred_vel.device),
                                                mask[:, :, :, 1:])
            
            loss_terms["loss"] = loss_terms.get('vb', 0.) + \
                            loss_terms.get('loss_data', 0.) + \
                            loss_terms.get('loss_data_vel', 0.) + \
                            loss_terms.get('loss_geo_xyz', 0) + \
                            loss_terms.get('loss_geo_xyz_vel', 0) + \
                            loss_terms.get('loss_foot_contact', 0)
            
            return model_output.permute(0, 3, 1, 2), loss_terms
        
        return model_output.permute(0, 3, 1, 2)
        
    
    def evaluate_sampling(self, dataloader, save_folder_name):
        self.model.eval()
        self.model.training = False
        common.mkdir('%s/%s' % (self.save_dir, save_folder_name))
        
        datas = next(iter(dataloader)) 
        datas = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas.items()}
        cond = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas['conditions'].items()}
        x_start = datas['data']
        t, _ = self.schedule_sampler.sample(dataloader.batch_size, self.device)
        with torch.no_grad():
            model_output = self.diffuse(x_start, t, cond, noise=None, return_loss=False)
        
        common_past_motion = cond['past_motion'].permute(0, 3, 1, 2)
        self.export_samples(x_start, common_past_motion, '%s/%s/' % (self.save_dir, save_folder_name), 'gt')
        self.export_samples(model_output, common_past_motion, '%s/%s/' % (self.save_dir, save_folder_name), 'pred')
        
        self.logger.info(f'Evaluate the sampling {save_folder_name} at epoch {self.epoch}')
        

    def export_samples(self, future_motion_feature, past_motion_feature, save_path, prefix):
        motion_feature = torch.cat((past_motion_feature, future_motion_feature), dim=1)
        rotations = nn_transforms.repr6d2quat(motion_feature[:, :, :-1]).cpu().numpy()
        root_pos = motion_feature[:, :, -1, :3].cpu().numpy()
        
        for samplie_idx in range(future_motion_feature.shape[0]):
            T_pose_template = self.dataloader.dataset.T_pose.copy()
            T_pose_template.rotations = rotations[samplie_idx]
            T_pose_template.positions = np.zeros((rotations[samplie_idx].shape[0], T_pose_template.positions.shape[1], T_pose_template.positions.shape[2]))
            T_pose_template.positions[:, 0] = root_pos[samplie_idx]
            T_pose_template.export(f'{save_path}/motion_{samplie_idx}.{prefix}.bvh', save_ori_scal=True)  
        