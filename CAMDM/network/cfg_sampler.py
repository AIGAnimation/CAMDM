# This code is based on https://github.com/GuyTevet/motion-diffusion-model
import torch.nn as nn
from copy import deepcopy

# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model, config):
        super().__init__()
        self.model = model  # model is the actual model to run

        assert config.trainer.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'

        # pointers to inner model
        self.rot_req = self.model.rot_req
        self.rot_feat_dim = self.model.rot_feat_dim
        self.joint_num = self.model.joint_num
        self.clip_len = self.model.clip_len
        self.input_feats = self.model.input_feats
        self.local_cond = self.model.local_cond
        self.global_cond = self.model.global_cond

        self.latent_dim = self.model.latent_dim
        self.ff_size = self.model.ff_size
        self.num_layers = self.model.num_layers
        self.num_heads = self.model.num_heads
        self.dropout = self.model.dropout
        self.ablation = self.model.ablation
        self.activation = self.model.activation
        self.clip_dim = self.model.clip_dim

    def forward(self, x, timesteps, y=None):
        y_uncond = deepcopy(y)
        y_uncond['uncond'] = True
        out = self.model(x, timesteps, y)
        out_uncond = self.model(x, timesteps, y_uncond)
        return out_uncond + (y['scale'].view(-1, 1, 1) * (out - out_uncond))
