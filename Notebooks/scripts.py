# %% [markdown]
# # Measure Pitch, HNR, Jitter, Shimmer, Formants, and Estimate VTL

# %% [markdown]
# ### The extracted features
# * voiceID
# * duration
# * meanF0Hz
# * stdevF0Hz
# * HNR
# * localJitter
# * localabsoluteJitter
# * rapJitter
# * ppq5Jitter
# * ddpJitter
# * localShimmer
# * localdbShimmer
# * apq3Shimmer
# * apq5Shimmer
# * apq11Shimmer
# * ddaShimmer
# * f1_mean
# * f2_mean
# * f3_mean
# * f4_mean
# * f1_median
# * f2_median
# * f3_median
# * f4_median
# * JitterPCA
# * ShimmerPCA
# * pF
# * fdisp
# * avgFormant
# * mff
# * fitch_vtl
# * delta_f
# * vtl_delta_f

# %% [markdown]
# ## Import the external modules

# %%
#!/usr/bin/env python3
import glob
import numpy as np
import pandas as pd
import parselmouth 
import statistics
import librosa
import noisereduce as nr
import scipy.stats as stats
from concurrent.futures import ThreadPoolExecutor
import scipy

from joblib import Parallel, delayed
from tqdm import tqdm
from pydub import AudioSegment
from parselmouth.praat import call
from scipy.stats.mstats import zscore
from scipy.stats import mode as scipy_mode
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool
import pandas as pd
import warnings
import glob
import os
warnings.filterwarnings("ignore")

# %% [markdown]
# ## This function measures duration, pitch, HNR, jitter, and shimmer

# %%
# This is the function to measure source acoustics using default male parameters.

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
    
    return duration, meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer

# %% [markdown]
# ## This function measures formants at each glottal pulse
# 
# Puts, D. A., Apicella, C. L., & Cárdenas, R. A. (2012). Masculine voices signal men's threat potential in forager and industrial societies. Proceedings of the Royal Society of London B: Biological Sciences, 279(1728), 601-609.
# 
# Adapted from: DOI 10.17605/OSF.IO/K2BHS

# %%
# This function measures formants using Formant Position formula
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
    
    return f1_mean, f2_mean, f3_mean, f4_mean, f1_median, f2_median, f3_median, f4_median


# %% [markdown]
# ## This function runs a 2-factor Principle Components Analysis (PCA) on Jitter and Shimmer

# %%
def runPCA(df):
    # z-score the Jitter and Shimmer measurements
    measures = ['localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter',
                'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 'apq11Shimmer', 'ddaShimmer']
    x = df.loc[:, measures].values
    x = StandardScaler().fit_transform(x)
    # PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['JitterPCA', 'ShimmerPCA'])
    principalDf
    return principalDf

# %% [markdown]
# ## Preprocessing to the audio file

# %%
def remove_silence(audio):
    unsilenced = []
    time_intervals = librosa.effects.split(audio, top_db=25, ref=np.max).tolist()
    for start, end in time_intervals:
        unsilenced += audio.tolist()[start:end+1]
    unsilenced = np.array(unsilenced)

    return unsilenced

def normalize(audio): 
    rms = np.sqrt(np.mean(audio**2))
    current_db = 20 * np.log10(rms)
    target_db = -20.0
    gain = target_db - current_db
    audio_normalized = audio * (10**(gain / 20))
    return audio_normalized

# %% [markdown]
# ## Enhanced parallel feature extraction

# %%
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

columns = [
    'voiceID', 'duration', 'meanF0Hz', 'stdevF0Hz', 'HNR', "IQR", "sd", "sfm", "Q25", "sp.ent", "mode",
    'localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter',
    'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 'apq11Shimmer', 'ddaShimmer',
    'f1_mean', 'f2_mean', 'f3_mean', 'f4_mean',
    'f1_median', 'f2_median', 'f3_median', 'f4_median'
]

BATCH_SIZE = 5000



# Function to extract features from a single audio file
def extract_features(file_path):
    try:    
        # print(f"Processing {file_path}...")
        # Load and preprocess the audio
        audio, sr = librosa.load('../Data/'+file_path, sr=None, mono=True)
        audio = remove_silence(audio)
        audio = normalize(audio)
        audio = nr.reduce_noise(y=audio, sr=sr)
        sound = parselmouth.Sound(audio, sampling_frequency=sr)

        # Extract features
        duration, meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, \
        localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer = measurePitch(
            audio, sr, sound, 75, 300, "Hertz"
        )

        f1_mean, f2_mean, f3_mean, f4_mean, f1_median, f2_median, f3_median, f4_median = measureFormants(
            sound, file_path, 75, 300
        )

        special = measureSpecialFeatures(audio, sr)

        f = {
            'voiceID': file_path,
            'duration': duration,
            'meanF0Hz': meanF0,
            'stdevF0Hz': stdevF0,
            'HNR': hnr,
            'localJitter': localJitter,
            'localabsoluteJitter': localabsoluteJitter,
            'rapJitter': rapJitter,
            'ppq5Jitter': ppq5Jitter,
            'ddpJitter': ddpJitter,
            'localShimmer': localShimmer,
            'localdbShimmer': localdbShimmer,
            'apq3Shimmer': apq3Shimmer,
            'apq5Shimmer': aqpq5Shimmer,
            'apq11Shimmer': apq11Shimmer,
            'ddaShimmer': ddaShimmer,
            'f1_mean': f1_mean,
            'f2_mean': f2_mean,
            'f3_mean': f3_mean,
            'f4_mean': f4_mean,
            'f1_median': f1_median,
            'f2_median': f2_median,
            'f3_median': f3_median,
            'f4_median': f4_median
        }

        f.update(special)
        return f

    except Exception as e:
        # print(f"Failed to process {file_path}: {e}")
        return None

# Parallel processing of audio files
def process_and_save_in_batches(audio_files, type):
    for i in range(0, len(audio_files), BATCH_SIZE):
        batch_files = audio_files[i:i + BATCH_SIZE]
        res = None
        with Pool(processes=os.cpu_count()) as pool:
            res = list(tqdm(pool.imap(extract_features, batch_files), total=len(batch_files)))
        filtered_res = [item for item in res if item is not None]
        df = pd.DataFrame(filtered_res)
        df.to_csv(f"../CSVs/{type}_{i // BATCH_SIZE+2}.csv", index=False)
        print(f"Saved batch {i // BATCH_SIZE} with {len(df)} entries.")
        del df  # Free memory

if __name__ == "__main__":
    males_audio_files =list(pd.read_csv('males2.csv')['path'])
    females_audio_files = list(pd.read_csv('females2.csv')['path'])

    # print(males_audio_files[2])
    process_and_save_in_batches(males_audio_files,"males_general")
    process_and_save_in_batches(females_audio_files,"females_general")

