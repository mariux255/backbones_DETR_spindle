import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class u_net_backbone(nn.Module):
    def __init__(self):
        super().__init__()
        n_groups = 8

        # DOWNSAMPLING

        #self.pad = nn.Conv1d(1, 1, kernel_size = 1, stride = 2)
        
        
        # ENCODER GROUND LEVEL (LEVEL 1)
        self.conv_1_1 = nn.Conv1d(1, 16, kernel_size = 5, dilation = 2, padding = 'same')
        self.batch_1_1 = nn.BatchNorm1d(16)

        self.conv_1_2 = nn.Conv1d(16, 16, kernel_size = 5, dilation = 2, padding = 'same')
        self.batch_1_2 = nn.BatchNorm1d(16)

        # LEVEL 2
        self.pool_1 = nn.MaxPool1d(kernel_size = 4)

        self.conv_2_1 = nn.Conv1d(16, 32, kernel_size = 5, dilation = 2, padding = 'same')
        self.batch_2_1 = nn.BatchNorm1d(32)

        self.conv_2_2 = nn.Conv1d(32, 32, kernel_size = 5, dilation = 2, padding = 'same')
        self.batch_2_2 = nn.BatchNorm1d(32)
        self.drop_2_1 = nn.Dropout(0.3)


        # ENCODER BOTTOM LEVEL
        self.pool_2 = nn.MaxPool1d(kernel_size = 4)

        self.conv_3_1 = nn.Conv1d(32, 64, kernel_size = 5, dilation = 2, padding = 'same')
        self.batch_3_1 = nn.BatchNorm1d(64)
        self.drop_3_1 = nn.Dropout(0.5)
        
        self.conv_3_2 = nn.Conv1d(64, 64, kernel_size = 5, dilation = 2, padding = 'same')
        self.batch_3_2 = nn.BatchNorm1d(64)
        self.drop_3_2 = nn.Dropout(0.5)

        # DECODER LEVEL 2
        # UPSAMPLING
        self.upsample_2 = nn.Upsample(scale_factor = 4, mode = 'nearest')
        self.conv_2_3 = nn.Conv1d(64, 32, kernel_size = 4, dilation = 1, padding = 'same')
        

        # 
        self.conv_2_4 = nn.Conv1d(64, 32, kernel_size = 5, dilation = 1, padding = 'same')
        self.batch_2_4 = nn.BatchNorm1d(32)
        

        self.conv_2_5 = nn.Conv1d(32, 32, kernel_size = 5, dilation = 1, padding = 'same')
        self.batch_2_5 = nn.BatchNorm1d(32)
        self.drop_2_2 = nn.Dropout(0.3)
        
        # DECODER GROUND LEVEL (LEVEL 1)
        # UPSAMPLING
        self.upsample_1 = nn.Upsample(scale_factor = 4, mode = 'nearest')
        self.conv_1_3 = nn.Conv1d(32, 16, kernel_size = 4, dilation = 1, padding = 'same')

        # 
        self.conv_1_4 = nn.Conv1d(32, 16, kernel_size = 5, dilation = 1, padding = 'same')
        self.batch_1_4 = nn.BatchNorm1d(16)

        self.conv_1_5 = nn.Conv1d(16, 16, kernel_size = 5, dilation = 1, padding = 'same')
        self.batch_1_5 = nn.BatchNorm1d(16)


        self.conv_1_6 = nn.Conv1d(16, 2, kernel_size = 1, dilation = 1)
        
    def forward(self, tensor_list):
        #print(tensor_list.shape)
        # DOWNSAMPLING
        #downsampled_input = self.downsample(tensor_list)
        #extrapolation = int(np.ceil(tensor_list.shape[1] / (4*4*4)) * (4*4*4) - tensor_list.shape[1])
        padded_input = F.pad(tensor_list, (2, 2), mode='reflect')
        #padded_input = tensor_list

        # GROUND LEVEL FORWARD
        level_1 = self.batch_1_1(F.relu(self.conv_1_1(padded_input)))
        level_1 = self.batch_1_2(F.relu(self.conv_1_2(level_1)))

        # POOLING AND LEVEL 2
        level_1_down = self.pool_1(level_1)
        level_2 = self.batch_2_1(F.relu(self.conv_2_1(level_1_down)))
        level_2 = self.drop_2_1(self.batch_2_2(F.relu(self.conv_2_2(level_2))))

        # POOLING AND BOTTOM LEVEL
        level_2_down = self.pool_2(level_2)
        level_3 = self.drop_3_1(self.batch_3_1(F.relu(self.conv_3_1(level_2_down))))
        level_3 = self.drop_3_2(self.batch_3_2(F.relu(self.conv_3_2(level_3))))

        # UPSAMPLING AND FEATURE FUSION (LEVEL 2)
        level_3_upsampled = self.upsample_2(level_3)
        level_2_up = self.conv_2_3(level_3_upsampled)
        
        #print(level_2.shape)
        #print(level_3_upsampled.shape)
        dec_level_2 = torch.cat((level_2, level_2_up), 1)

        dec_level_2 = self.drop_2_2(self.batch_2_4(F.relu(self.conv_2_4(dec_level_2))))
        dec_level_2 = self.batch_2_5(F.relu(self.conv_2_5(dec_level_2)))

        # UPSAMPLING AND FEATURE FUSION (UPPER LEVEL)
        level_2_upsampled = self.upsample_1(dec_level_2)
        level_1_up = self.conv_1_3(level_2_upsampled)
        
        dec_level_1 = torch.cat((level_1, level_1_up), 1)

        dec_level_1 = self.batch_1_4(F.relu(self.conv_1_4(dec_level_1)))
        dec_level_1 = self.batch_1_5(F.relu(self.conv_1_5(dec_level_1)))

        dec_level_1 = F.relu(self.conv_1_6(dec_level_1))


        diff = dec_level_1.shape[2] - tensor_list.shape[2]
        crop_dims = [diff // 2, diff // 2 + diff % 2]

        if crop_dims[1] == 0:
            dec_level_1 = dec_level_1[:, :, crop_dims[0]:]
        else:
            dec_level_1 = dec_level_1[:, :, crop_dims[0]:-crop_dims[1]]

        # Not used when calculating loss
        #smooth = F.avg_pool1d(dec_level_1, 42, stride=1)

        return dec_level_1
