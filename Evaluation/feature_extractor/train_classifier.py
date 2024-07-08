import os
import numpy as np
import pickle
import sys
sys.path.append('./')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from tqdm import tqdm

import scipy.ndimage.filters as filters
from scipy.spatial.transform import Rotation as R

import torch.nn.functional as F

window_size = 60

class MotionDataset(Dataset):
    def __init__(self, pkl_path, clip_len, clip_offset=1):        
        self.local_position_list = []
        self.style_name_list, self.style_index_list = [], []
        
        self.clip_len = clip_len
        
        data_source = pickle.load(open(pkl_path, 'rb'))
        item_frame_indices_list = []
        for data_idx, data_item in enumerate(tqdm(data_source)):
            
            local_position = np.array(data_item['local_position'], dtype=np.float32)
            style_name, style_idx = data_item['style_name'], data_item['style_idx']
            
            frame_num = local_position.shape[0]
            
            clip_indices = np.arange(0, frame_num - clip_len + 1, clip_offset)[:, None] + np.arange(clip_len)
            clip_indices_with_idx = np.hstack((np.full((len(clip_indices), 1), data_idx, dtype=clip_indices.dtype), clip_indices))
            
            self.local_position_list.append(local_position.reshape(frame_num, -1))
            item_frame_indices_list.append(clip_indices_with_idx)
            self.style_index_list.append(style_idx)
            self.style_name_list.append(style_name)
        
        self.style_name_list = sorted(list(set(self.style_name_list)))
        self.item_frame_indices = np.concatenate(item_frame_indices_list, axis=0)
        self.njoints = local_position.shape[1]
                
        print("Dataset size: %d clips with %s frames per clip" % (self.__len__(), clip_len))
        
        
    def __len__(self):
        return self.item_frame_indices.shape[0]
    
    def __getitem__(self, index):
        item_frame_indice = self.item_frame_indices[index]
        motion_idx, frame_indices = item_frame_indice[0], item_frame_indice[1:]
        
        locao_position = self.local_position_list[motion_idx][frame_indices]
        style_idx = self.style_index_list[motion_idx]
        
        return locao_position, style_idx


class Classifier(nn.Module):
    def __init__(self, in_feat, class_num, device):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_feat, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.residual1x1_1 = nn.Conv1d(in_channels=in_feat, out_channels=64, kernel_size=1)  # 1x1 conv for first residual connection
        self.residual1x1_2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)  # 1x1 conv for second residual connection
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(in_features=128 * window_size, out_features=1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(in_features=512, out_features=class_num)
        self.data_mean = None
        self.data_std = None        


    def forward(self, x):
        feats = self.get_feats(x)
        out = self.fc3(feats)
        return feats, out


    def get_feats(self, x):
        residual = self.residual1x1_1(x)  # adjust channels of residual to match out
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out += residual  # Residual connection after adjusting channels
        residual = self.residual1x1_2(out)  # adjust channels of residual to match out
        out = F.leaky_relu(self.bn2(self.conv2(out)))
        out += residual
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        out = F.leaky_relu(self.bn3(self.fc1(out)))
        out = F.leaky_relu(self.bn4(self.fc2(out)))
        return out 


if __name__ == '__main__':

    file_path = 'data/100style_position_ori+ybot.pkl'
              
    motion_dataset = MotionDataset(file_path, clip_len=window_size, clip_offset=10)
    motion_dataloader = torch.utils.data.DataLoader(motion_dataset, batch_size=1024, shuffle=True, num_workers=4)

    feature_dim = motion_dataset.njoints * 3
    
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    model = Classifier(in_feat=feature_dim, class_num=len(motion_dataset.style_name_list), device=device)
    model.to(device)
    
    # model = torch.nn.DataParallel(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epoch_num = 50
    
    for epoch in range(epoch_num):
        epoch_loss = []
        correct, total = 0, 0
        for i, data in enumerate(motion_dataloader, 0):
            local_positions, style_idx = data[0].to(device), data[1].to(device)
            
            inputs = local_positions.permute(0, 2, 1)
            optimizer.zero_grad()
            feats, out = model(inputs)
            loss = criterion(out, style_idx)             
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(out.data, 1)
            total += style_idx.size(0)
            correct += (predicted == style_idx).sum().item()
            epoch_loss.append(loss.item())
            
        print('epoch %d loss: %.3f' % (epoch + 1, np.mean(epoch_loss)))
        print('Accuracy of the network on the training data for this epoch: %05f%%' % (100 * correct / total))
    
    save_dic = {
        'state_dict': model.state_dict(),
        'in_feat': feature_dim,
        'acc': 100 * correct / total,
        'class_names': motion_dataset.style_name_list,
    }
    
    torch.save(save_dic, './100style_position_classifier.pth')
    