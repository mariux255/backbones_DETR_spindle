import numpy as np
import json
import os
import scipy.io
import mne
from scipy.signal import butter, resamply_poly, sosfiltfilt
from scipy.signal.windows import hann
from scipy.fft import rfft, rfftfreq


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


def butter_bandpass_filter(data, lowcut, highcut, sample_rate, order):
    """
    Bandpass filter the data using Butterworth IIR filters.

    Two digital Butterworth IIR filters with the specified order are created, one highpass filter for the lower critical
    frequency and one lowpass filter for the higher critical frequency. Both filters use second-order sections (SOS).
    Then first the highpass filter is applied on the given data and on its result the lowpass filter is applied.
    Both filters are applied as forward-backward digital filters to correct the non-linear phase.

    Parameters
    ----------
    data : ndarray
        The data to be filtered; format (n_samples,)
    lowcut : float
        The lower critical frequency
    highcut : float
        The higher critical frequency
    sample_rate : float
        The sampling rate of the given data
    order : int
        The order of the used filters

    Returns
    -------
    data : ndarray
        the bandpass filtered data; format (n_samples,)
    """

    sos_high = butter(order, lowcut, btype='hp', fs=sample_rate, output='sos')
    sos_low = butter(order, highcut, btype='lp', fs=sample_rate, output='sos')
    return sosfiltfilt(sos_low, sosfiltfilt(sos_high, data, padlen=3 * order), padlen=3 * order)


def downsample(data, sample_rate, resampling_frequency):
    """
    Downsample the given data to a target frequency.

    Uses the scipy resample_poly function to transform the data from the original sample_rate to resampling_frequency.

    Parameters
    ----------
    data : ndarray
        The data to be downsampled; format (n_samples,)
    sample_rate : int or float
        The original sample rate of data
    resampling_frequency : int or float
        The target sample rate to transform data into, must not be higher than sample_rate

    Returns
    -------
    data : ndarray
        The downsampled data; format (n_samples_new,)
    """

    if (sample_rate != int(sample_rate)) | (resampling_frequency != int(resampling_frequency)):
        raise Exception('parameters "sample_rate" and "resampling_frequency" have to be integers')
    elif sample_rate < resampling_frequency:
        raise Exception('the original sample frequency must not be lower than the resample frequency')
    elif sample_rate == resampling_frequency:
        return data

    sample_rate = int(sample_rate)
    resampling_frequency = int(resampling_frequency)

    gcd = np.gcd(sample_rate, resampling_frequency)

    up = resampling_frequency // gcd
    down = sample_rate // gcd

    return resample_poly(data, up, down)

def sample_to_win(data_sample, win_length_sec, win_step_sec, sample_rate):
    n_samples = data_sample.shape[0]

    # Length of the window as number of samples
    n_samples_per_win = round(win_length_sec * sample_rate)
    # Length of the window step as number of samples
    n_samples_per_win_step = round(win_step_sec * sample_rate)
    # Total length of the time series; in seconds
    data_length_sec = n_samples / sample_rate
    # Number of windows, last one might be incomplete
    n_windows = np.ceil((data_length_sec - win_length_sec) / win_step_sec).astype(int) + 1

    # 2d array with dimensions (n_windows,n_samples_per_win) representing the indices of the sliding window
    indexer = np.arange(n_samples_per_win)[None, :] + n_samples_per_win_step * np.arange(n_windows)[:, None]

    # Make sure a possible incomplete window at the end is handled correctly
    # Therefore temporarily for indexer>=n_samples just repeat the last value...
    data_win = data_sample[np.minimum(indexer, n_samples - 1)]
    # ...and afterwards mask theses values before returning
    return np.ma.masked_array(data_win, indexer >= n_samples)


def win_to_sample(data_win, win_length_sec, win_step_sec, sample_rate, n_samples):
    n_windows = data_win.shape[0]

    # Length of the window as number of samples
    n_samples_per_win = round(win_length_sec * sample_rate)
    # The number of samples overlapping between two windows
    overlap = np.ceil(win_length_sec / win_step_sec).astype(int)

    data_sample = np.full((overlap, n_samples), np.nan)
    i = 0
    while i < n_windows:
        # Iterate over the overlapping windows, effectively iterating the currently considered window (see "i = i + 1" below)
        for j in range(overlap):
            # The indices have to be computed here, as i is modified within this inner loop
            # Start index of the currently considered window
            idx_start = round(i * win_step_sec * sample_rate)
            # Stop index of the currently considered window
            idx_stop = idx_start + n_samples_per_win

            if i < n_windows:
                if idx_stop < n_samples:
                    # Complete window
                    data_sample[j, idx_start:idx_stop] = data_win[i]
                else:
                    # Incomplete window
                    data_sample[j, idx_start:] = data_win[i]

            # Incremented within the inner loop!
            i = i + 1

    # Mask all remaining NaN values (which always occur at beginning and end of the samples)
    return np.ma.masked_array(data_sample, np.isnan(data_sample))


def power_spectral_density(data, win_length_sec, win_step_sec, sample_rate, zero_pad_sec):
    # Length of the FFT window as number of samples
    n_samples_per_fft_win = round(win_length_sec * sample_rate)
    # Length of the FFT window padded to zero_pad_sec as number of samples
    n_samples_per_padded_fft_win = round(zero_pad_sec * sample_rate)

    # Hann window to scale/smooth the signal before FFT
    win_hann_coeffs = hann(n_samples_per_fft_win)
    # Noise gain
    ng = np.square(win_hann_coeffs).mean()

    # Get matrix of sliding PSD windows
    windows = sample_to_win(data, win_length_sec, win_step_sec, sample_rate)
    # Calculate the mean for every window
    windows_mean = windows.mean(axis=1)

    # Remove DC offset from windows...
    windows = windows - np.c_[windows_mean]
    # and scale them using the Hann window
    windows_scaled = windows * win_hann_coeffs

    # Perform the FFT with the window zero padded to zero_pad_sec. As the input is always real, only the positive
    # frequency terms are considered (as the result of FFT is Hermitian-symmetric), so the rfft function is used
    # As windows_scaled is a masked array and the rfft function cannot handle these, the masked values are filled with
    # zeros to not influence the result of rfft
    fft_modules = np.abs(rfft(windows_scaled.filled(0), n_samples_per_padded_fft_win))
    # Get the frequency bins used by the rfft function
    freq_bins = rfftfreq(n_samples_per_padded_fft_win, 1 / sample_rate)

    # Apply the IntegSpectPow PSA normalization as described by Hanspeter Schmid
    fft_modules = np.square(fft_modules / n_samples_per_padded_fft_win)
    fft_modules[:, 1:] = fft_modules[:, 1:] * 2
    psd = fft_modules / ng

    # As the DC offset was removed early, it needs to be added as first frequency again
    psd[:, 0] = windows_mean

    # Return the PSD and the used frequency bins
    return psd, freq_bins


def baseline_windows(data_per_win, win_length_sec, win_step_sec, bsl_length_sec):
    n_windows = data_per_win.shape[0]
    # Number of windows per baseline (with bsl_length_sec), last one (per baseline) might be incomplete
    n_windows_per_bsl = np.ceil((bsl_length_sec - win_length_sec) / win_step_sec).astype(int) + 1

    # The baseline per sliding window value is usually centered around the value, but always bsl_length_sec long
    # Therefore the start index of every baseline is its index minus half the number of windows per baseline...
    bsl_win_start_idx = np.arange(n_windows) - np.ceil((n_windows_per_bsl - 1) / 2).astype(int)
    # ... but always at least 0 and at most n_windows - n_windows_per_bsl
    bsl_win_start_idx = np.minimum(np.maximum(bsl_win_start_idx, 0), n_windows - n_windows_per_bsl)
    # The stop index (excluded) is now simply the start index plus n_windows_per_bsl
    bsl_win_stop_idx = bsl_win_start_idx + n_windows_per_bsl

    # As a "multidimensional" call to np.arange() isn't possible, the following lines produce the same output as
    # "bsl_wins_idx = np.array([np.arange(bsl_win_start_idx[i], bsl_win_stop_idx[i]) for i in range(n_windows)])"
    # see https://codereview.stackexchange.com/questions/83018/vectorized-numpy-version-of-arange-with-multiple-start-stop
    bsl_wins_idx = np.repeat(bsl_win_stop_idx - np.arange(1, n_windows + 1) * n_windows_per_bsl,
                             n_windows_per_bsl) + np.arange(n_windows * n_windows_per_bsl)
    bsl_wins_idx = bsl_wins_idx.reshape(n_windows, -1)

    # Return the baseline windows per value in data_per_win with format (n_windows,n_windows_per_bsl)
    return data_per_win[bsl_wins_idx]


def baseline_z_score(data_per_win, win_length_sec, win_step_sec, bsl_length_sec):
    # The baseline windows for every sliding window value (so one baseline per sliding window)
    bsl_per_win = baseline_windows(data_per_win, win_length_sec, win_step_sec, bsl_length_sec)

    # Only consider values in the baseline included in the 10th-90th percentile, all other values are set to NaN
    limits = np.percentile(data_per_win, [10, 90])
    bsl_per_win[(bsl_per_win < limits[0]) | (bsl_per_win > limits[1])] = np.nan

    # Calculate the mean and std per baseline window, ignoring the NaN values
    bsl_per_win_mean = np.nanmean(bsl_per_win, axis=1)
    bsl_per_win_std = np.nanstd(bsl_per_win, axis=1)

    # Transform every sliding window value to its z-score using the corresponding baseline window
    return (data_per_win - bsl_per_win_mean) / bsl_per_win_std


def unmask_result(result):
    mask = result.mask
    # Make sure all samples have a value
    if np.any(mask):
        # If there are still NaN values, use linear interpolation
        # see https://stackoverflow.com/a/9537830
        result[mask] = np.interp(np.nonzero(mask)[0], np.nonzero(~mask)[0], result[~mask])

    # Return the result as a normal ndarray (not masked)
    return result.compressed()


def absolute_power_values(data, win_length_sec, win_step_sec, sample_rate):
    n_samples = data.shape[0]

    # Filter the sigma signal as bandpass filter from 11 to 16 Hz
    sigma_data = butter_bandpass_filter(data, 11, 16, sample_rate, 20)
    # Get matrix of sliding windows
    win_sample_matrix = sample_to_win(sigma_data, win_length_sec, win_step_sec, sample_rate)
    # Calculate average squared power per window
    absolute_power_per_win = np.square(win_sample_matrix).mean(axis=1)

    # Calculate absolute sigma power per sample (multiple values for samples with overlapping windows)
    absolute_power_per_sample = win_to_sample(absolute_power_per_win, win_length_sec, win_step_sec, sample_rate,
                                              n_samples)
    # Return the average absolute sigma power per sample log10 transformed
    return unmask_result(np.log10(absolute_power_per_sample.mean(axis=0)))


def relative_power_values(data, win_length_sec, win_step_sec, bsl_length_sec, sample_rate):
    n_samples = data.shape[0]

    # The time on which the sliding windows should be zero padded before performing the FFT
    zero_pad_sec = 2
    # Sliding window PSD and the used frequency bins
    psd, freq_bins = power_spectral_density(data, win_length_sec, win_step_sec, sample_rate, zero_pad_sec)

    # Calculate the sigma power by summing the PSD windows in the sigma band
    # As freq_index_stop should be excluded, 1 is added
    freq_index_start, freq_index_stop = np.argmin(np.abs(freq_bins - 11)), np.argmin(np.abs(freq_bins - 16)) + 1
    psd_sigma_freq = psd[:, freq_index_start:freq_index_stop].sum(axis=1)

    # Calculate the total power by summing the PSD windows in the broadband signal excluding delta band
    # As freq_index_stop should be excluded, 1 is added
    freq_index_start, freq_index_stop = np.argmin(np.abs(freq_bins - 4.5)), np.argmin(np.abs(freq_bins - 30)) + 1
    psd_total_freq = psd[:, freq_index_start:freq_index_stop].sum(axis=1)

    # Calculate the relative ratio of sigma power and total power log10 transformed
    relative_power_per_win = np.log10(psd_sigma_freq / psd_total_freq)

    # Calculate the z-score of every relative power using a baseline window
    relative_power_per_win_z_score = baseline_z_score(relative_power_per_win, win_length_sec, win_step_sec,
                                                      bsl_length_sec)

    # Calculate relative power ratio per sample (multiple values for samples with overlapping windows)
    relative_power_per_sample = win_to_sample(relative_power_per_win_z_score, win_length_sec, win_step_sec, sample_rate,
                                              n_samples)

    # Return the average relative power ratio per sample
    return unmask_result(relative_power_per_sample.mean(axis=0))


def covariance_values(data, win_length_sec, win_step_sec, bsl_length_sec, sample_rate):
    n_samples = data.shape[0]

    # Filter the sigma signal as bandpass filter from 11 to 16 Hz
    sigma_data = butter_bandpass_filter(data, 11, 16, sample_rate, 20)

    # Get matrix of sliding windows for broadband signal
    win_sample_matrix_raw = sample_to_win(data, win_length_sec, win_step_sec, sample_rate)
    # Get matrix of sliding windows for sigma band
    win_sample_matrix_sigma = sample_to_win(sigma_data, win_length_sec, win_step_sec, sample_rate)

    n_windows = win_sample_matrix_raw.shape[0]

    # Calculate the covariance between the two signals for every window except the last one
    covariance_per_win = np.array(
        [np.cov(win_sample_matrix_raw[i], win_sample_matrix_sigma[i])[0, 1] for i in range(n_windows - 1)])
    # The last window might be incomplete, so use the np.ma.cov function which handles missing data, but is much
    # slower than the np.cov function
    covariance_last_win = np.ma.cov(win_sample_matrix_raw[-1], win_sample_matrix_sigma[-1])[0, 1]
    covariance_per_win = np.r_[covariance_per_win, covariance_last_win]

    # Covariance between the two signals log10 transformed
    covariance_per_win_no_negative = covariance_per_win
    covariance_per_win_no_negative[covariance_per_win_no_negative < 0] = 0
    covariance_per_win_log10 = np.log10(covariance_per_win + 1)

    # Calculate the z-score of every covariance using a baseline window
    covariance_per_win_z_score = baseline_z_score(covariance_per_win_log10, win_length_sec, win_step_sec,
                                                  bsl_length_sec)

    # Calculate covariance per sample (multiple values for samples with overlapping windows)
    covariance_per_sample = win_to_sample(covariance_per_win_z_score, win_length_sec, win_step_sec, sample_rate,
                                          n_samples)

    # Return the average covariance per sample
    return unmask_result(covariance_per_sample.mean(axis=0))


def correlation_values(data, win_length_sec, win_step_sec, sample_rate):
    n_samples = data.shape[0]

    # Filter the sigma signal as bandpass filter from 11 to 16 Hz
    sigma_data = butter_bandpass_filter(data, 11, 16, sample_rate, 20)

    # Get matrix of sliding windows for broadband signal
    win_sample_matrix_raw = sample_to_win(data, win_length_sec, win_step_sec, sample_rate)
    # Get matrix of sliding windows for sigma band
    win_sample_matrix_sigma = sample_to_win(sigma_data, win_length_sec, win_step_sec, sample_rate)

    n_windows = win_sample_matrix_raw.shape[0]

    # Calculate the correlation between the two signals for every window except the last one
    correlation_per_win = np.array(
        [np.corrcoef(win_sample_matrix_raw[i], win_sample_matrix_sigma[i])[0, 1] for i in range(n_windows - 1)])
    # The last window might be incomplete, so use the np.ma.corrcoef function which handles missing data, but is much
    # slower than the np.corrcoef function
    correlation_last_win = np.ma.corrcoef(win_sample_matrix_raw[-1], win_sample_matrix_sigma[-1])[0, 1]
    correlation_per_win = np.r_[correlation_per_win, correlation_last_win]

    # Calculate correlation per sample (multiple values for samples with overlapping windows)
    correlation_per_sample = win_to_sample(correlation_per_win, win_length_sec, win_step_sec, sample_rate, n_samples)

    # Return the average correlation per sample
    return unmask_result(correlation_per_sample.mean(axis=0))


def possible_spindle_indices(features, thresholds):
    # All four features must be above their respective thresholds at least once (simultaneously) during a spindle
    all_thresholds_exceeded_idx = np.where(np.all(features > thresholds, axis=1))[0]

    # Only the features a7_relative_sigma_power and a7_sigma_correlation (features indices [0,2]) exceeding their
    # respective thresholds are relevant for the start and end of a spindle detection (as long as all features exceed
    # their respective thresholds at least once simultaneously during the spindle)
    absolute_thresholds_exceeded = np.all(features[:, [0, 2]] > thresholds[[0, 2]], axis=1)

    # The changes of the features a7_relative_sigma_power and a7_sigma_correlation exceeding their thresholds are calculated
    # The .astype(int) conversion leads to 1 for True and 0 for False
    # The leading padded zero allows detecting a spindle that starts at the first sample, the trailing padded zero
    # allows detecting a spindle that ends at the last sample
    absolute_threshold_changes = np.diff(np.r_[0, absolute_thresholds_exceeded.astype(int), 0])

    # If a sample is the start of a spindle, there is a change from False to True (0 to 1) for
    # "features[:,[0,2]] > thresholds[[0,2]]", which means the value of the diff function has to be 1
    start_candidates_idx = np.where(absolute_threshold_changes == 1)[0]

    # If a sample is the first sample after an end of a spindle, there is a change from True to False (1 to 0) for
    # "features[:,[0,2]] > thresholds[[0,2]]", which means the value of the diff function has to be -1
    stop_candidates_idx = np.where(absolute_threshold_changes == -1)[0]

    # The (start_candidates_idx,stop_candidates_idx) pairs are only actual starts and stops of spindles, if there is at
    # least one sample in between at which all four thresholds are exceeded simultaneously
    start_stop_candidates_idx = np.c_[start_candidates_idx, stop_candidates_idx]
    start_stop_idx = np.array([[start, stop] for start, stop in start_stop_candidates_idx if
                               np.any((start <= all_thresholds_exceeded_idx) & (all_thresholds_exceeded_idx < stop))])
    # In case no spindle candidate is found, the dimension is (1, 0) and has to be set to (0, 2)
    start_stop_idx = start_stop_idx.reshape(-1, 2)

    # Return the (included) start indices and the (excluded) stop indices
    return start_stop_idx[:, 0], start_stop_idx[:, 1]


def detect_spindles(data, thresholds, win_length_sec, win_step_sec, sample_rate):
    bsl_length_sec = 30
    spindle_length_min_sec = 0.3
    spindle_length_max_sec = 2.5

    # Calculate the four features of the given data
    a7_absolute_sigma_power = absolute_power_values(data, win_length_sec, win_step_sec, sample_rate)
    a7_relative_sigma_power = relative_power_values(data, win_length_sec, win_step_sec, bsl_length_sec, sample_rate)
    a7_sigma_covariance = covariance_values(data, win_length_sec, win_step_sec, bsl_length_sec, sample_rate)
    a7_sigma_correlation = correlation_values(data, win_length_sec, win_step_sec, sample_rate)

    # Stack the features to a (n_samples, 4) matrix
    features = np.stack((a7_absolute_sigma_power, a7_relative_sigma_power, a7_sigma_covariance, a7_sigma_correlation),
                        axis=1)
    # With the features and the given thresholds calculate the start and stop indices of possible spindles
    start_idx, stop_idx = possible_spindle_indices(features, thresholds)

    spindle_length = stop_idx - start_idx
    # Only detected spindles whose length (in seconds) is between spindle_length_min_sec and spindle_length_max_sec are considered
    valid_idx = (spindle_length >= spindle_length_min_sec * sample_rate) & (
            spindle_length <= spindle_length_max_sec * sample_rate)

    # Only return the indices of valid spindles
    start_idx, stop_idx = start_idx[valid_idx], stop_idx[valid_idx]
    return features, np.c_[start_idx, stop_idx]

def A7(x, sr, return_features=False):
    thresholds = np.array([1.25, 1.3, 1.3, 0.69])
    win_length_sec = 0.3
    win_step_sec = 0.1
    features, spindles = detect_spindles(x, thresholds, win_length_sec, win_step_sec, sr)
    return spindles / sr if not return_features else (spindles / sr, features)

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
