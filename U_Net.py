import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class u_net_backbone(nn.Module):
    def __init__(self):
        super().__init__()
        n_groups = 8
        
        # ENCODER GROUND LEVEL (LEVEL 1)
        self.conv_1_1 = nn.Conv1d(1, 64, kernel_size = 5, stride = 2, dilation = 2)
        self.batch_1_1 = nn.GroupNorm(n_groups, 64)

        self.conv_1_2 = nn.Conv1d(64, 128, kernel_size = 5, dilation = 2)
        self.batch_1_2 = nn.GroupNorm(n_groups, 128)


        # ENCODER BOTTOM LEVEL
        self.pool_1 = nn.MaxPool1d(kernel_size = 4)

        self.conv_2_1 = nn.Conv1d(128, 256, kernel_size = 5, dilation = 2, padding = 'same')
        self.batch_2_1 = nn.GroupNorm(n_groups, 256)

        self.conv_2_2 = nn.Conv1d(256, 256, kernel_size = 5, dilation = 2, padding = 'same')
        self.batch_2_2 = nn.GroupNorm(n_groups, 256)

        
        # DECODER GROUND LEVEL (LEVEL 1)
        # UPSAMPLING
        self.upsample_1 = nn.Upsample(scale_factor = 4, mode = 'nearest')
        self.conv_1_3 = nn.Conv1d(256, 128, kernel_size = 4, dilation = 1, padding = 'same')

        # 
        self.conv_1_4 = nn.Conv1d(256, 128, kernel_size = 5, dilation = 1)
        self.batch_1_4 = nn.GroupNorm(n_groups, 128)

        self.conv_1_5 = nn.Conv1d(128, 128, kernel_size = 5, dilation = 1)
        self.batch_1_5 = nn.GroupNorm(n_groups, 128)


        self.conv_1_6 = nn.Conv1d(128, 2, kernel_size = 1, dilation = 1)
        

        self.num_channels = 128
    def forward(self, tensor_list: NestedTensor):
        # GROUND LEVEL FORWARD
        level_1 = self.batch_1_1(F.relu(self.conv_1_1(tensor_list.tensors)))
        level_1 = self.batch_1_2(F.relu(self.conv_1_2(level_1)))

        # POOLING AND BOTTOM LEVEL
        level_1_down = self.pool_1(level_1)
        level_2 = self.batch_2_1(F.relu(self.conv_2_1(level_1_down)))
        level_2 = self.batch_2_2(F.relu(self.conv_2_2(level_2)))

        # UPSAMPLING AND FEATURE FUSION
        level_2_upsampled = self.upsample_1(level_2)
        level_1_up = self.conv_1_3(level_2_upsampled)
        
        dec_level_1 = torch.cat((level_1, level_1_up), 1)

        dec_level_1 = self.batch_1_4(F.relu(self.conv_1_4(dec_level_1)))
        dec_level_1 = self.batch_1_5(F.relu(self.conv_1_5(dec_level_1)))

        dec_level_1 = self.conv_1_6(dec_level_1)

        smooth = F.avg_pool1d(dec_level_1, 42, stride=1)

        return smooth
