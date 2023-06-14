


def load_data(input_path = '/home/marius/Documents/THESIS/data/1D_MASS_MODA_processed/input', label_path = '/home/marius/Documents/THESIS/data/1D_MASS_MODA_processed/labels'):
    
    


    input_path = input_path
    label_path = label_path
    input_dict = {}
    label_dict = {}
    temp_input_list = []
    temp_output_list = []
    for root, dirs, files in os.walk(input_path):
        for name in files:
            if name.endswith('npy'):
                temp_input_list.append(os.path.join(root, name))
                input_dict[int(name[:-4])] = os.path.join(root, name)

    for root, dirs, files in os.walk(label_path):
        for name in files:
            if name.endswith('json'):
                temp_output_list.append(os.path.join(root, name))
                label_dict[int(name[:-5])] = os.path.join(root, name)

    master_path_list = []

    for in_path in temp_input_list:
        for la_path in temp_output_list:
            if in_path[-16:-3] == la_path[-17:-4]:
                master_path_list.append((in_path,la_path))

    return master_path_list

def pred_stats(outputs, targets):
    
    # Loop through batches to compute F1 score through training.

    
    F1_list = []
    temp_tp = 0
    total_spindle_count = 0
    total_pred_count = 0

    target_bbox = targets['boxes']

    TP = 0

    target_bbox = np.asarray(target_bbox)
    total_spindle_count += target_bbox.shape[0]
    total_pred_count += len(outputs)
    for k in range(target_bbox.shape[0]):
        tar_box = target_bbox[k,:]
        tar_box_start = tar_box[0] - tar_box[1]/2
        tar_box_end = tar_box[0] + tar_box[1]/2

        best_match = -1

        if len(outputs) == 0:
            continue

        for j,out_box in enumerate(outputs):
            out_box_start = out_box[0] - out_box[1]/2
            out_box_end = out_box[0] + out_box[1]/2

            #if ((out_box_end > tar_box_start) and (out_box_start <= tar_box_start)):
            if iou(out_box, tar_box) > iou(outputs[best_match], tar_box):
                best_match = j
        if iou(outputs[best_match],tar_box) > 0.2:
            TP +=1
        


    #F1_list = np.asarray(F1_list)
    #print("F1 MEAN:", np.mean(F1_list), " F1 STD:", np.std(F1_list), " TP:", temp_tp, " FP:", FP, " Number of spindles:", total_spindle_count)
    
    return (TP, total_pred_count, total_spindle_count)

def f1_calculate(model, device, dataloader):
    TP = 0
    total_pred_count = 0
    total_spindle_count = 0
    for samples, targets in dataloader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        temp_tp, temp_pred_count, temp_spindle_count = pred_stats(outputs, targets)
        TP += temp_tp
        total_pred_count += temp_pred_count
        total_spindle_count += temp_spindle_count
    
    f1 = f1_score(TP, total_pred_count, total_spindle_count)

    print("F1 score:", f1, " True positives:", TP, " Total predictions:", total_pred_count, " Total spindles:", total_spindle_count)

    return (f1, TP, total_pred_count, total_spindle_count)
        


def f1_score(TP, total_pred_count, total_spindle_count):
    
    FP = total_pred_count - TP
    FN = total_spindle_count - TP
        
    if (TP + FP) == 0:
        PRECISION = TP
    else:
        PRECISION = (TP)/(TP + FP)
        
    RECALL = (TP)/(TP+FN)

    if (PRECISION + RECALL) == 0:
            F1 = 0
    else:
         F1 = (2 * PRECISION * RECALL)/(PRECISION + RECALL)

    return F1



def iou(out,tar):
    out_box_start = out[0] - out[1]/2
    out_box_end = out[0] + out[1]/2

    tar_box_start = tar[0] - tar[1]/2
    tar_box_end = tar[0] + tar[1]/2

    overlap_start = max(out_box_start, tar_box_start)
    overlap_end = min(out_box_end, tar_box_end)
    union_start = min(out_box_start, tar_box_start)
    union_end = max(out_box_end, tar_box_end)

    return ((overlap_end - overlap_start)/(union_end-union_start))

def overlap(out, tar, threshold):
    out_box_start = out[0] - out[1]/2
    out_box_end = out[0] + out[1]/2

    tar_box_start = tar[0] - tar[1]/2
    tar_box_end = tar[0] + tar[1]/2

    overlap_start = max(out_box_start, tar_box_start)
    overlap_end = min(out_box_end, tar_box_end)
    union_start = min(out_box_start, tar_box_start)
    union_end = max(out_box_end, tar_box_end)

    if (overlap_end - overlap_start) >= (threshold * (tar_box_end-tar_box_start)):
        return True
    else:
        return False



def main(master_path_list):
    TP = 0
    total_pred_count = 0
    total_spindle_count = 0


    for seq in master_path_list:
        food, labels = seq
        
        f = open(labels)
        labels = (json.load(f))
        f.close()
        
        eeg = np.load(food)
        sr = 256
        eeg = downsample(butter_bandpass_filter(eeg, 0.3, 30.0, sr, 10), sr, 100)
        eeg = eeg * (10**6)
        spindles = A7(eeg,100)
        spindles = spindles/(len(eeg)/100)
        spindles[:,1] = spindles[:,1] - spindles[:,0]
        spindles[:,0] = spindles[:,0] + (spindles[:,1]/2)
        guesses_to_keep = []
        for i, spindle in enumerate(spindles):
            if not (spindle[1]*115 < 0.3):
                guesses_to_keep.append(spindle)
        spindles = np.asarray(guesses_to_keep)
        temp_tp, temp_pred_count, temp_spindle_count = pred_stats(spindles, labels)
        TP += temp_tp
        total_pred_count += temp_pred_count
        total_spindle_count += temp_spindle_count
        
    f1 = f1_score(TP, total_pred_count, total_spindle_count)

    print("F1 score:", f1, " True positives:", TP, " Total predictions:", total_pred_count, " Total spindles:", total_spindle_count)

master_path_list = load_data()
main(master_path_list)