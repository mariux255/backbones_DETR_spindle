import torch
from scipy.signal import butter, sosfilt, sosfreqz
import scipy.io
import random
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import matplotlib.image as mpimg
from torchvision.io import read_image
import json
import cv2


class MODA_proc(Dataset):
    def __init__(self, input_path = '/scratch/s174411/center_width/1D_MASS_MODA_processed/input/', label_path = '/scratch/s174411/center_width/1D_MASS_MODA_processed/labels/'):
        self.input_path = input_path
        self.label_path = label_path
        self.input_dict = {}
        self.label_dict = {}
        temp_input_list = []
        temp_output_list = []
        for root, dirs, files in os.walk(self.input_path):
            for name in files:
                if name.endswith('npy'):
                    temp_input_list.append(os.path.join(root, name))
                    self.input_dict[int(name[:-4])] = os.path.join(root, name)

        for root, dirs, files in os.walk(self.label_path):
            for name in files:
                if name.endswith('json'):
                    temp_output_list.append(os.path.join(root, name))
                    self.label_dict[int(name[:-5])] = os.path.join(root, name)

        self.master_path_list = []
        
        for in_path in temp_input_list:
            for la_path in temp_output_list:
                if in_path[-16:-3] == la_path[-17:-4]:
                    self.master_path_list.append((in_path,la_path))
                


    def __len__(self):
        return len(self.master_path_list)

    def __getitem__(self, idx):
        #print(self.input_dict[idx])
        model_input, labels = self.master_path_list[idx]
        eeg_input = np.load(model_input)
        # Standardize
        eeg_input = (eeg_input - np.mean(eeg_input))/np.std(eeg_input)

        eeg_input = torch.FloatTensor(eeg_input)
        eeg_input = eeg_input[None, :]

        #print('dataloader shape')

        #image = np.array(cv2.imread(self.input_dict[idx]))
        #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        #image = read_image(self.input_dict[idx])
        f = open(labels)
        
        labels = (json.load(f))
        f.close()


        input_length = int(256*30/2)
        sumo_label_format = np.zeros(input_length)
        for bbox in labels['boxes']:
            
            box_start = bbox[0] - bbox[1]/2
            box_end = bbox[0] + bbox[1]/2
            box_start_scaled = int(box_start * input_length)
            box_end_scaled = int(box_end * input_length)
            sumo_label_format[box_start_scaled:box_end_scaled] = 1

        spindle = False
        for sample in sumo_label_format:
            if sample == 1:
                spindle = True

        if not spindle:
            print("found")
        #print(type(sumo_label_format))
        sumo_label_format = torch.FloatTensor(sumo_label_format)
        #print(sumo_label_format.shape)

        return eeg_input, sumo_label_format