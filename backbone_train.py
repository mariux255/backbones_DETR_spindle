from U_Net import u_net_backbone
from simple_cnn import conv_simple
from MODA_dataloader import MODA_proc
from dice_loss import GeneralizedDiceLoss

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



# Loading data, setting up GPU use, setting up variables for model training
def main(BATCH_SIZE = 128, EPOCHS = 40):
    dataset_train = MODA_proc(input_path = '/scratch/s174411/train_val_test_30/TRAIN/input/', label_path = '/scratch/s174411/train_val_test_30/TRAIN/labels/')
    dataset_val = MODA_proc(input_path = '/scratch/s174411/train_val_test_30/VAL/input/', label_path = '/scratch/s174411/train_val_test_30/VAL/labels/')

    data_loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    data_loader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)




    net = u_net_backbone()
    if torch.cuda.is_available():
        device = "cuda:0"
        # if torch.cuda.device_count() > 1:
        #     print(torch.cuda.device_count(), " GPU's GIVEN")
        #     net = nn.DataParallel(net)
        # else:
        #     print(device)
        net.to(device)

    criterion = GeneralizedDiceLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay = 0.03)

    training_loss = []
    validation_loss = []
    f1_df_train = pd.DataFrame()
    f1_df_val = pd.DataFrame()
    for j, epoch in enumerate(range(EPOCHS)):  # loop over the dataset multiple times
        net.train()

        running_loss = []
        total_TP = 0
        total_pred_count = 0
        total_spindle_count = 0
        for i, batch in enumerate(data_loader_train):
            
                
            model_input, labels = batch
            model_input = model_input.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(model_input)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())

            if (epoch % 1 == 0):
                TP, pred_count, spindle_count = get_metrics(outputs, labels)
                total_TP += TP
                total_pred_count += pred_count
                total_spindle_count += spindle_count

        print(f"EPOCH:{epoch}")
        print("TRAINING")
        print("Loss: ", round(sum(running_loss)/len(running_loss), 6))
        training_loss.append(sum(running_loss)/len(running_loss))

        
        
        if (epoch % 1 == 0):
            F1 = f1_score(total_TP, total_pred_count, total_spindle_count)
            print("F1 MEAN:", round(F1, 6), " TP:", total_TP, " Total predictions:", total_pred_count,
                " Number of spindles:", total_spindle_count)
            row = {'F1': F1, 'TP': total_TP, 'Total pred': total_pred_count, 'Total spindle': total_spindle_count}
            f1_df_train = f1_df_train.append(row, ignore_index = True)

        net.eval()
        
        running_loss = []
        print("VALIDATION")
        running_loss = []
        total_TP = 0
        total_pred_count = 0
        total_spindle_count = 0
        for i, batch in enumerate(data_loader_val): 
            model_input, labels = batch
            model_input = model_input.to(device)
            labels = labels.to(device)

            outputs = net(model_input)

            loss = criterion(outputs, labels)


            running_loss.append(loss.item())


            if (epoch % 1 == 0):
                TP, pred_count, spindle_count = get_metrics(outputs, labels)
                total_TP += TP
                total_pred_count += pred_count
                total_spindle_count += spindle_count


        print("Loss: ", round(sum(running_loss)/len(running_loss), 6))
        if (epoch % 1 == 0):
            F1 = f1_score(total_TP, total_pred_count, total_spindle_count)
            print("F1 MEAN:", round(F1, 6), " TP:", total_TP, " Total predictions:", total_pred_count,
                " Number of spindles:", total_spindle_count)
            row = {'F1': F1, 'TP': total_TP, 'Total pred': total_pred_count, 'Total spindle': total_spindle_count}
            f1_df_val = f1_df_val.append(row, ignore_index = True)
        print("")
        
        validation_loss.append(sum(running_loss)/len(running_loss))
    training_loss = np.asarray(training_loss)
    validation_loss = np.asarray(validation_loss)
    np.savetxt('/home/s174411/code/backbones_DETR_spindle/logs/loss_train.csv', training_loss, delimiter=",")
    np.savetxt('/home/s174411/code/backbones_DETR_spindle/logs/loss_val.csv', validation_loss, delimiter=",")
    f1_df_train.to_csv('/home/s174411/code/backbones_DETR_spindle/logs/f1_train.csv', index=False)
    f1_df_val.to_csv('/home/s174411/code/backbones_DETR_spindle/logs/f1_val.csv', index=False)    
    torch.save(net, '/home/s174411/code/backbones_DETR_spindle/logs/sumo.pt')
    torch.save(net.state_dict(), '/home/s174411/code/backbones_DETR_spindle/logs/sumo_state_dict.pt')

def out_to_vector(output):
    moving_avg = 42
    s = moving_avg - 1
    vector = F.pad(output, (s // 2, s // 2 + s % 2), mode='constant', value=0)

    vector_smoothed =  F.avg_pool1d(vector, moving_avg, stride=1)
    
    vector_softmax = F.softmax(vector_smoothed, dim=0)

    return vector_softmax.argmax(dim=0).long()


def vector_to_spindle_list(vector, debug = False):
    
    prev_class = 0
    list_of_spindles = []
    vector = vector.numpy()
    for i, instance_class in enumerate(vector):
        if (instance_class == 1 and prev_class == 1):
            prev_class = 1
            if (i+1 == len(vector)):
                spindle.append(i)
                list_of_spindles.append(spindle)
            continue

        if (instance_class == 1 and prev_class == 0):
            spindle = []
            spindle.append(i)
            prev_class = 1
            continue

        if (instance_class == 0 and prev_class == 1):
            spindle.append(i)
            list_of_spindles.append(spindle)
            prev_class = 0
            continue

        prev_class = 0
    return list_of_spindles
        
    #print(vector)
def refine_spindle_list(list_of_spindles, target = False):
    refined_spindle_list = []
    for spindle in list_of_spindles:
        start = spindle[0]/(100)
        end = spindle[1]/(100)
        if ((end-start) < 0.3 and not target):
            continue
        else:
            refined_spindle_list.append((start, end))

    return refined_spindle_list


def get_metrics(outputs, targets):
    
    # Loop through batches to compute F1 score through training
    temp_tp = 0
    total_spindle_count = 0
    total_pred_count = 0
    TP = 0

    for i in range(outputs.shape[0]):

        pred = out_to_vector(outputs[i,:,:].cpu())

        pred_spindles = vector_to_spindle_list(pred)
        #print(len(pred_spindles))

        #pred_spindles = refine_spindle_list(pred_spindles)
        #print(len(pred_spindles)

        target = targets[i]
        t_spindles = vector_to_spindle_list(target.cpu())
        #t_spindles = refine_spindle_list(t_spindles)

        total_spindle_count += len(t_spindles)
        batch_spindle_count = len(t_spindles)

        # if len(t_spindles) == 0:
        #     spindle = False
        #     for l, sample in enumerate(target):
        #         if sample == 1:
        #             spindle = True
        #     if spindle:
        #         print('not found')
        #         print(vector_to_spindle_list(target.cpu(), debug = True))
        #         print(len(target))

        batch_pred_count = len(pred_spindles)
        total_pred_count += len(pred_spindles)
        for k in range(len(t_spindles)):
            tar_box = t_spindles[k]
            #print(tar_box)
            
            best_match = -1

            if len(pred_spindles) == 0:
                continue
            
            for j,out_box in enumerate(pred_spindles):

                if overlap(out_box, tar_box) > overlap(pred_spindles[best_match], tar_box):
                    best_match = j
            #print(pred_spindles[best_match])
            #print(tar_box)
            if overlap(pred_spindles[best_match],tar_box) >= 0.2:
                TP +=1
            
    return (TP, total_pred_count, total_spindle_count)

def f1_score(TP, total_pred_count, total_spindle_count):

    FP = total_pred_count - TP
    FN = total_spindle_count - TP
        
    if (TP + FP) == 0:
        PRECISION = TP
    else:
        PRECISION = (TP)/(TP + FP)
        
    RECALL = (TP)/(TP+FN)

    if (PRECISION + RECALL) == 0:
        return 0
    else:
        return(2 * PRECISION * RECALL)/(PRECISION + RECALL)
        
def iou(out,tar):
    out_box_start = out[0]
    out_box_end = out[1]

    tar_box_start = tar[0]
    tar_box_end = tar[1]

    overlap_start = max(out_box_start, tar_box_start)
    overlap_end = min(out_box_end, tar_box_end)
    union_start = min(out_box_start, tar_box_start)
    union_end = max(out_box_end, tar_box_end)

    return ((overlap_end - overlap_start)/(union_end-union_start))

def overlap(out, tar):
    out_box_start = out[0]
    out_box_end = out[1]

    tar_box_start = tar[0]
    tar_box_end = tar[1]

    overlap_start = max(out_box_start, tar_box_start)
    overlap_end = min(out_box_end, tar_box_end)
    union_start = min(out_box_start, tar_box_start)
    union_end = max(out_box_end, tar_box_end)

    return (overlap_end - overlap_start) / (tar_box_end-tar_box_start)


        

main()
#torch.save(net, '/home/marius/Documents/OneDrive/MSc/StartUP/Code/m1_stats_features.pt')
print('Finished Training')