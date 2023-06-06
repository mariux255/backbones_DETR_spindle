import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class conv_simple(nn.Module):
    def __init__(self):
        super().__init__()
        n_groups = 8
        
        # ENCODER GROUND LEVEL (LEVEL 1)
        self.conv_1 = nn.Conv1d(1, 16, kernel_size = 5, dilation = 2, padding = 'same')
        self.batch_1 = nn.BatchNorm1d(16)

        self.conv_2 = nn.Conv1d(16, 32, kernel_size = 5, dilation = 2, padding = 'same')
        self.batch_2 = nn.BatchNorm1d(32)

        self.conv_3 = nn.Conv1d(32, 16, kernel_size = 5, dilation = 2, padding = 'same')
        self.batch_3 = nn.BatchNorm1d(16)

      
        
    def forward(self, tensor_list):
        #print(tensor_list.shape)
        # DOWNSAMPLING
        #downsampled_input = self.downsample(tensor_list)
        #extrapolation = int(np.ceil(tensor_list.shape[1] / (4*4*4)) * (4*4*4) - tensor_list.shape[1])
        padded_input = F.pad(tensor_list, (2, 2), mode='reflect')

        features = self.batch_1(F.relu(self.conv_1(padded_input)))
        features = self.batch_2(F.relu(self.conv_2(features)))

        features = self.batch_3(F.relu(self.conv_3(features)))


        diff = features.shape[2] - tensor_list.shape[2]
        crop_dims = [diff // 2, diff // 2 + diff % 2]

        if crop_dims[1] == 0:
            features = features[:, :, crop_dims[0]:]
        else:
            features = features[:, :, crop_dims[0]:-crop_dims[1]]

        # Not used when calculating loss
        #smooth = F.avg_pool1d(dec_level_1, 42, stride=1)

        return features
