import numpy as np
import librosa
from pydub import AudioSegment
import numpy as np
import librosa
from multiprocessing import Pool
import warnings
import parselmouth
from scipy.stats import skew, kurtosis
from parselmouth.praat import call
import statistics
from scipy.stats import mode as scipy_mode
import scipy


warnings.filterwarnings("ignore")

def unsilenced_audio(audio, top_db=25):
    intervals = librosa.effects.split(audio, top_db=top_db, ref=np.max)
    return np.concatenate([audio[start:end] for start, end in intervals])

def normalize_loudness(audio, target_db=-20.0):
    rms = np.sqrt(np.mean(audio ** 2))
    if rms == 0:
        return audio
    current_db = 20 * np.log10(rms)
    gain = target_db - current_db
    return audio * (10 ** (gain / 20))

# def load_audio(filepath):
    # ext = filepath.split('.')[-1].lower()
    
    # if ext == 'mp3':
    #     audio = AudioSegment.from_mp3(filepath)
    #     samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    #     samples /= np.iinfo(audio.array_type).max
        
    #     if audio.frame_rate != sr:
    #         samples = librosa.resample(samples, orig_sr=audio.frame_rate, target_sr=sr)
    # else:
    #     samples, _ = librosa.load(filepath, sr=sr)
    
    # return samples

def measurePitch(audio, sr, sound, f0min, f0max, unit):
    sound = parselmouth.Sound(sound) # read the sound
    duration = call(sound, "Get total duration") # duration
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
    meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
    stdevF0 = call(pitch, "Get standard deviation", 0 ,0, unit) # get standard deviation
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    return {
        "duration": duration,
        "meanF0": meanF0,
        "stdevF0": stdevF0,
        "hnr": hnr,
        "localJitter": localJitter,
        "localabsoluteJitter": localabsoluteJitter,
        "rapJitter": rapJitter,
        "ppq5Jitter": ppq5Jitter,
        "ddpJitter": ddpJitter,
        "localShimmer": localShimmer,
        "localdbShimmer": localdbShimmer,
        "apq3Shimmer": apq3Shimmer,
        "aqpq5Shimmer": aqpq5Shimmer,
        "apq11Shimmer": apq11Shimmer,
        "ddaShimmer": ddaShimmer
    }

def measureFormants(sound, wave_file, f0min,f0max):
    sound = parselmouth.Sound(sound) # read the sound
    pitch = call(sound, "To Pitch (cc)", 0, f0min, 15, 'no', 0.03, 0.45, 0.01, 0.35, 0.14, f0max)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    
    formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
    numPoints = call(pointProcess, "Get number of points")

    f1_list = []
    f2_list = []
    f3_list = []
    f4_list = []
    
    # Measure formants only at glottal pulses
    for point in range(0, numPoints):
        point += 1
        t = call(pointProcess, "Get time from index", point)
        f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        f4 = call(formants, "Get value at time", 4, t, 'Hertz', 'Linear')
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)
        f4_list.append(f4)
    
    f1_list = [f1 for f1 in f1_list if str(f1) != 'nan']
    f2_list = [f2 for f2 in f2_list if str(f2) != 'nan']
    f3_list = [f3 for f3 in f3_list if str(f3) != 'nan']
    f4_list = [f4 for f4 in f4_list if str(f4) != 'nan']
    
    # calculate mean formants across pulses
    f1_mean = statistics.mean(f1_list)
    f2_mean = statistics.mean(f2_list)
    f3_mean = statistics.mean(f3_list)
    f4_mean = statistics.mean(f4_list)
    
    # calculate median formants across pulses, this is what is used in all subsequent calcualtions
    # you can use mean if you want, just edit the code in the boxes below to replace median with mean
    f1_median = statistics.median(f1_list)
    f2_median = statistics.median(f2_list)
    f3_median = statistics.median(f3_list)
    f4_median = statistics.median(f4_list)
    
    return {
        "f1_mean": f1_mean,
        "f2_mean": f2_mean,
        "f3_mean": f3_mean,
        "f4_mean": f4_mean,
        "f1_median": f1_median,
        "f2_median": f2_median,
        "f3_median": f3_median,
        "f4_median": f4_median
    }

def measureSpecialFeatures(y, sr):
    n_fft = 2048  # Specify FFT window size
    stft = np.abs(librosa.stft(y, n_fft=n_fft))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Normalize STFT to obtain a probability distribution over frequency bins
    stft_norm = stft / (np.sum(stft, axis=0, keepdims=True) + 1e-6)
    
    # Mean frequency per frame
    meanfreq = np.sum(freqs[:, None] * stft_norm, axis=0)

    # Spectral flatness over all frames
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    mean_spectral_flatness = np.mean(spectral_flatness)  # ✔ fix: use all frames

    # Spectral entropy (normalized power spectrum used)
    power_spectrum = stft**2
    psd_norm = power_spectrum / (np.sum(power_spectrum, axis=0, keepdims=True) + 1e-10)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10), axis=0)
    mean_spectral_entropy = np.mean(spectral_entropy)  # ✔ fix: correct normalization

    # IQR across frequency bins per frame, averaged
    iqr_per_frame = scipy.stats.iqr(stft, axis=0)
    avg_iqr = np.mean(iqr_per_frame) / 1000  # ✔ fix: compute IQR across freqs, not meanfreq

    # Standard deviation of mean frequency
    std_meanfreq = np.std(meanfreq) / 1000  # ✔ same, just confirmed correct

    # 25th percentile of mean frequency
    q25 = np.percentile(meanfreq, 25) / 1000  # ✔ same, just confirmed correct

    # Mode frequency from peak per frame
    mode_freq = freqs[np.argmax(stft, axis=0)]
    mode_result = scipy_mode(mode_freq, keepdims=True)
    mode_frequency = float(mode_result.mode[0] / 1000)  # ✔ fix: clean usage of scipy_mode

    return {
        "IQR": float(avg_iqr),
        "sd": float(std_meanfreq),
        "sfm": float(mean_spectral_flatness),
        "Q25": float(q25),
        "sp.ent": float(mean_spectral_entropy),
        "mode": mode_frequency
    }

def extract_pause_and_phoneme_features(audio, sr, top_db=25):
    """
    Extracts pause duration, pause counts, and duration of phonemes from the given audio.
    
    Parameters:
        audio (numpy.ndarray): The audio signal.
        sr (int): The sampling rate of the audio.
        top_db (int): The threshold (in decibels) below reference to consider as silence.

    Returns:
        dict: A dictionary containing pause duration, pause counts, and phoneme duration.
    """
    # Detect non-silent intervals
    intervals = librosa.effects.split(audio, top_db=top_db, ref=np.max)
    
    # Calculate pause durations
    pause_durations = []
    for i in range(1, len(intervals)):
        pause_start = intervals[i - 1][1]
        pause_end = intervals[i][0]
        pause_durations.append((pause_end - pause_start) / sr)
    
    # Calculate phoneme durations
    phoneme_durations = [(end - start) / sr for start, end in intervals]
    
    # Aggregate features
    total_pause_duration = sum(pause_durations)
    total_phoneme_duration = sum(phoneme_durations)
    pause_count = len(pause_durations)
    
    return {
        "total_pause_duration": total_pause_duration,
        "pause_count": pause_count,
        "total_phoneme_duration": total_phoneme_duration
    }


def extract_f0_features(audio, sr):
    """
    Extracts F0-related features (mean, variance, std, min, max, skewness, kurtosis) from the audio signal.

    Parameters:
        audio (numpy.ndarray): The audio signal.
        sr (int): The sampling rate of the audio.

    Returns:
        dict: A dictionary containing F0-related features.
    """
    try:
        # Create a Parselmouth Sound object
        sound = parselmouth.Sound(audio, sampling_frequency=sr)
        
        # Extract pitch (F0) using Parselmouth
        pitch = sound.to_pitch()
        f0_values = pitch.selected_array['frequency']
        
        # Filter out unvoiced frames (F0 = 0)
        f0_values = f0_values[f0_values > 0]
        
        if len(f0_values) == 0:
            # If no voiced frames are found, return zeros for all features
            return {
                "meanF0": 0,
                "varF0": 0,
                "stdF0": 0,
                "minF0": 0,
                "maxF0": 0,
                "skewnessF0": 0,
                "kurtosisF0": 0
            }
        
        # Calculate F0 features
        mean_f0 = np.mean(f0_values)
        var_f0 = np.var(f0_values)
        std_f0 = np.std(f0_values)
        min_f0 = np.min(f0_values)
        max_f0 = np.max(f0_values)
        skewness_f0 = skew(f0_values)
        kurtosis_f0 = kurtosis(f0_values)
        
        return {
            "meanF0": mean_f0,
            "varF0": var_f0,
            "stdF0": std_f0,
            "minF0": min_f0,
            "maxF0": max_f0,
            "skewnessF0": skewness_f0,
            "kurtosisF0": kurtosis_f0
        }
    except Exception as e:
        # Handle any errors during F0 extraction
        return {
            "meanF0": 0,
            "varF0": 0,
            "stdF0": 0,
            "minF0": 0,
            "maxF0": 0,
            "skewnessF0": 0,
            "kurtosisF0": 0
        }
    
def calculate_speaking_rate(audio, sr):
    """
    Calculates the speaking rate (phonemes per second) after unsilencing the audio.

    Parameters:
        audio (numpy.ndarray): The audio signal.
        sr (int): The sampling rate of the audio.

    Returns:
        float: The speaking rate (phonemes per second).
    """
    # Remove silent parts of the audio
    unsilenced = unsilenced_audio(audio)
    
    # Total duration of the unsilenced audio in seconds
    total_unsilenced_duration = len(unsilenced) / sr
    
    # Detect non-silent intervals to estimate phoneme count
    intervals = librosa.effects.split(audio, top_db=25, ref=np.max)
    phoneme_count = len(intervals)  # Approximate phoneme count as the number of intervals
    
    # Calculate speaking rate
    if total_unsilenced_duration > 0:
        speaking_rate = phoneme_count / total_unsilenced_duration
    else:
        speaking_rate = 0  # Avoid division by zero
    return speaking_rate

def extract_intonation_features(audio, sr):
    """
    Extracts intonation features (pitch range, pitch variance, pitch contour) from the audio signal.

    Parameters:
        audio (numpy.ndarray): The audio signal.
        sr (int): The sampling rate of the audio.

    Returns:
        dict: A dictionary containing intonation features.
    """
    try:
        # Create a Parselmouth Sound object
        sound = parselmouth.Sound(audio, sampling_frequency=sr)
        
        # Extract pitch (F0) using Parselmouth
        pitch = sound.to_pitch()
        f0_values = pitch.selected_array['frequency']
        
        # Filter out unvoiced frames (F0 = 0)
        f0_values = f0_values[f0_values > 0]
        
        if len(f0_values) == 0:
            # If no voiced frames are found, return zeros for all features
            return {
                "pitch_range": 0,
                "pitch_variance": 0,
                "pitch_contour": []
            }
        
        # Calculate intonation features
        pitch_range = np.max(f0_values) - np.min(f0_values)
        pitch_variance = np.var(f0_values)
        pitch_contour = f0_values.tolist()  # Convert pitch contour to a list
        
        return {
            "pitch_range": pitch_range,
            "pitch_variance": pitch_variance,
            "pitch_contour": pitch_contour
        }
    except Exception as e:
        return {
            "pitch_range": 0,
            "pitch_variance": 0,
            "pitch_contour": []
        }


def extract_rhythm_features(audio, sr):
    """
    Extracts rhythm features (pause-to-speech ratio, duration variability) from the audio signal.

    Parameters:
        audio (numpy.ndarray): The audio signal.
        sr (int): The sampling rate of the audio.

    Returns:
        dict: A dictionary containing rhythm features.
    """
    try:
        # Detect non-silent intervals
        intervals = librosa.effects.split(audio, top_db=25, ref=np.max)
        
        # Calculate pause durations
        pause_durations = []
        for i in range(1, len(intervals)):
            pause_start = intervals[i - 1][1]
            pause_end = intervals[i][0]
            pause_durations.append((pause_end - pause_start) / sr)
        
        # Calculate phoneme durations
        phoneme_durations = [(end - start) / sr for start, end in intervals]
        
        # Aggregate rhythm features
        total_pause_duration = sum(pause_durations)
        total_phoneme_duration = sum(phoneme_durations)
        pause_to_speech_ratio = total_pause_duration / total_phoneme_duration if total_phoneme_duration > 0 else 0
        duration_variability = np.var(phoneme_durations) if phoneme_durations else 0
        
        return {
            "pause_to_speech_ratio": pause_to_speech_ratio,
            "duration_variability": duration_variability
        }
    except Exception as e:
        return {
            "pause_to_speech_ratio": 0,
            "duration_variability": 0
        }

def extract_mfcc_stft(audio, sr):
    """
    Extracts audio features including MFCCs, Spectral Centroid, Spectral Bandwidth, 
    Spectral Contrast, Spectral Flatness, Spectral Roll-off, and Zero Crossing Rate.
    Returns a concatenated feature vector with mean, median, and std statistics for each feature.
    """
    
    # Compute MFCCs and their statistics
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
    mfccs_stats = np.concatenate([
        np.mean(mfccs, axis=1),
        np.median(mfccs, axis=1),
        np.std(mfccs, axis=1)
    ])

    # STFT magnitude stats (optional to include)
    stft = librosa.stft(audio, n_fft=2048, hop_length=512)
    magnitude = np.abs(stft)
    stft_stats = np.concatenate([
        np.mean(magnitude, axis=1),
        np.median(magnitude, axis=1),
        np.std(magnitude, axis=1)
    ])

    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    centroid_stats = np.array([
        np.mean(spectral_centroid),
        np.median(spectral_centroid),
        np.std(spectral_centroid)
    ])

    # Spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    bandwidth_stats = np.array([
        np.mean(spectral_bandwidth),
        np.median(spectral_bandwidth),
        np.std(spectral_bandwidth)
    ])

    # Spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    contrast_stats = np.concatenate([
        np.mean(spectral_contrast, axis=1),
        np.median(spectral_contrast, axis=1),
        np.std(spectral_contrast, axis=1)
    ])

    # Spectral flatness
    spectral_flatness = librosa.feature.spectral_flatness(y=audio)
    flatness_stats = np.array([
        np.mean(spectral_flatness),
        np.median(spectral_flatness),
        np.std(spectral_flatness)
    ])

    # Spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)
    rolloff_stats = np.array([
        np.mean(spectral_rolloff),
        np.median(spectral_rolloff),
        np.std(spectral_rolloff)
    ])

    # Zero-crossing rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
    zcr_stats = np.array([
        np.mean(zero_crossing_rate),
        np.median(zero_crossing_rate),
        np.std(zero_crossing_rate)
    ])

    # Combine all features
    features = np.concatenate([
        mfccs_stats,
        stft_stats,
        centroid_stats,
        bandwidth_stats,
        contrast_stats,
        flatness_stats,
        rolloff_stats,
        zcr_stats
    ])

    return features

mfcc_columns = ['mfcc_mean_' + str(i) for i in range(1, 13 + 1)] + \
    ['mfcc_median_' + str(i) for i in range(1, 13 + 1)] + \
    ['mfcc_std_' + str(i) for i in range(1, 13 + 1)]

centroid_columns = ['centroid_mean', 'centroid_median', 'centroid_std']
bandwidth_columns = ['bandwidth_mean', 'bandwidth_median', 'bandwidth_std']

contrast_columns = ['contrast_mean_' + str(i) for i in range(1, 7 + 1)] + \
    ['contrast_median_' + str(i) for i in range(1, 7 + 1)] + \
    ['contrast_std_' + str(i) for i in range(1, 7 + 1)]

flatness_columns = ['flatness_mean', 'flatness_median', 'flatness_std']
rolloff_columns = ['rolloff_mean', 'rolloff_median', 'rolloff_std']
zcr_columns = ['zcr_mean', 'zcr_median', 'zcr_std']

# Combine all feature labels into one list
all_mfcc_columns = mfcc_columns + \
    centroid_columns + \
    bandwidth_columns + \
    contrast_columns + \
    flatness_columns + \
    rolloff_columns + \
    zcr_columns

def extract_stft_mean_std_median_IQR(audio, sr):
    """
    Extracts STFT features from the given audio and computes their mean, standard deviation, median, and interquartile range (IQR).
    Returns a concatenated feature vector.
    """
    # Compute the STFT
    stft = librosa.stft(audio, n_fft=2048, hop_length=512)
    
    # Convert to magnitude spectrogram
    magnitude = np.abs(stft)

    # Compute mean, standard deviation, median, and IQR for each frequency bin
    stft_mean = np.mean(magnitude, axis=1)
    stft_std = np.std(magnitude, axis=1)
    stft_median = np.median(magnitude, axis=1)
    
    # Concatenate the features into a single vector
    features = np.concatenate([stft_mean, stft_std,stft_median])

    return features

stft_cols=  ['stft_mean_' + str(i) for i in range(1, 14)] + \
            ['stft_std_' + str(i) for i in range(1, 14)] + \
            ['stft_median_' + str(i) for i in range(1, 14)]

# Function to extract features from a single audio file
def extract_features_preprocessed(file_path):
    try:

        audio, sr = librosa.load(file_path, sr=22050, mono=True)
        audio = normalize_loudness(audio)
        print("before parselmouth")
        sound = parselmouth.Sound(audio, sampling_frequency=sr)
        print("after parselmouth")

        # Extract pause and phoneme features
        features = extract_pause_and_phoneme_features(audio, sr)

        # Unsilence the audio
        unsilenced = unsilenced_audio(audio)
        
        # generals
        features.update(measurePitch(
            audio, sr, sound, 75, 300, "Hertz"
        ))
        features.update(measureFormants(
            sound, audio, 75, 300
        ))
        features.update(measureSpecialFeatures(
            audio, sr
        ))

        # Extract F0 features
        features.update(extract_f0_features(unsilenced, sr))
        
        # Extract intonation features
        features.update(extract_intonation_features(unsilenced, sr))
        
        # Extract rhythm features
        features.update(extract_rhythm_features(audio, sr))
        
        # Calculate speaking rate
        features["speaking_rate"] = calculate_speaking_rate(audio, sr)

        # Extract MFCCs and other features
        mfcc_stats = extract_mfcc_stft(audio, sr)
        mfcc_stats = list(mfcc_stats)

        f = { key: val for key, val in zip(all_mfcc_columns, mfcc_stats)}
        features.update(f)

        stft_stats = extract_stft_mean_std_median_IQR(audio, sr)
        stft_stats = list(stft_stats)

        f = {key: val for key, val in zip(stft_cols, stft_stats)}
        features.update(f)
        

        return features
    except Exception as e:
        return None

########################################################################################
#####################      transformations      #######################################
########################################################################################
def apply_log_column(column):
    min_val = column.min()
    shift = abs(min_val) + 1 if min_val <= 0 else 0
    return np.log(column + shift)

def apply_sqrt_column(column):
    min_val = column.min()
    shift = abs(min_val) + 1 if min_val < 0 else 0
    return np.sqrt(column + shift)

def apply_tranformations(df):
    df.drop(columns=['duration'], inplace=True)
    df.drop(columns=['pitch_contour'], inplace=True)

    df.fillna(0, inplace=True)

    log_features = [
    'stft_mean_10',
    'stft_mean_11'
    ]

    sqrt_features = [
        'stft_mean_12',
        'stft_mean_13',
        'stft_std_1',
        'mfcc_mean_9',
        'mfcc_std_5',
        'stdevF0',
        'localabsoluteJitter'
    ]
    
    for feature in log_features:
        df[feature+'_log'] = apply_log_column(df[feature])

    for feature in sqrt_features:
        df[feature+'_sqrt'] = apply_sqrt_column(df[feature])

    df.drop(columns=log_features + sqrt_features , inplace=True)