import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn


class MotionDiffusion(nn.Module):
    def __init__(self, input_feats, nstyles, njoints, nfeats, rot_req, clip_len,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.2,
                 ablation=None, activation="gelu", legacy=False, 
                 arch='trans_enc', cond_mask_prob=0, device=None):
        super().__init__()

        self.legacy = legacy
        self.training = True
        
        self.rot_req = rot_req
        self.nfeats = nfeats
        self.njoints = njoints
        self.clip_len = clip_len
        self.input_feats = input_feats

        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.ablation = ablation
        self.activation = activation
        self.cond_mask_prob = cond_mask_prob
        self.arch = arch

        # local conditions
        self.future_motion_process = MotionProcess(self.input_feats, self.latent_dim)
        self.past_motion_process = MotionProcess(self.input_feats, self.latent_dim)
        self.traj_trans_process = TrajProcess(2, self.latent_dim)
        self.traj_pose_process = TrajProcess(6, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        # global conditions
        self.embed_style = EmbedStyle(nstyles, self.latent_dim)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.arch == 'trans_enc':
            print("TRANS_ENC init")
            
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'trans_dec':
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=activation)
            self.seqEncoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                         num_layers=self.num_layers)

        elif self.arch == 'gru':
            print("GRU init")
            self.seqEncoder = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')
      
        self.output_process = OutputProcess(self.input_feats, self.latent_dim, self.njoints, self.nfeats)


    def forward(self, x, timesteps, past_motion, traj_pose, traj_trans, style_idx):
        bs, njoints, nfeats, nframes = x.shape
        
        time_emb = self.embed_timestep(timesteps)  # [1, bs, L]
        style_emb = self.embed_style(style_idx).unsqueeze(0)  # [1, bs, L]
        traj_trans_emb = self.traj_trans_process(traj_trans) # [N/2, bs, L] 
        traj_pose_emb = self.traj_pose_process(traj_pose) # [N/2, bs, L] 
        past_motion_emb = self.past_motion_process(past_motion)  # [past_frames, bs, L] 
        
        future_motion_emb = self.future_motion_process(x) 
        
        xseq = torch.cat((time_emb, style_emb, 
                          traj_trans_emb, traj_pose_emb,
                          past_motion_emb, future_motion_emb), axis=0)
        
        xseq = self.sequence_pos_encoder(xseq)
        output = self.seqEncoder(xseq)[-nframes:] 
        output = self.output_process(output)  
        return output
        

    def interface(self, x, timesteps, y=None):
        """
            x: [batch_size, frames, njoints, nfeats], denoted x_t in the paper 
            timesteps: [batch_size] (int)
            y: a dictionary containing conditions
        """
        bs, njoints, nfeats, nframes = x.shape
        
        style_idx = y['style_idx'] 
        past_motion = y['past_motion']
        traj_pose = y['traj_pose']
        traj_trans = y['traj_trans']
        
        # CFG on past motion
        keep_batch_idx = torch.rand(bs, device=past_motion.device) < (1-self.cond_mask_prob)
        past_motion = past_motion * keep_batch_idx.view((bs, 1, 1, 1))
        
        return self.forward(x, timesteps, past_motion, traj_pose, traj_trans, style_idx)
    

class MotionProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats) 
        x = self.poseEmbedding(x)  
        return x


class TrajProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs,  nfeats, nframes = x.shape
        x = x.permute((2, 0, 1))
        x = self.poseEmbedding(x)  
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class OutputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        output = self.poseFinal(output)  
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  
        return output
    
    
class EmbedStyle(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input.to(torch.long) 
        output = self.action_embedding[idx]
        return output
