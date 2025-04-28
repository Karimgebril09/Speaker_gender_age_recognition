import os
import librosa
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import warnings
import noisereduce as nr
warnings.filterwarnings("ignore")
from utils_audio import  unsilenced_audio, normalize_loudness, extract_age_features\
                        , extract_gender_features, extract_common_features

class DataProcessor:
    def __init__(self, directory, sr=None):
        self.directory = directory
        self.sr = sr
        self.audio_paths = []
        
    def load_all_data(self):
        files = os.listdir(self.directory)
        audio_files = [f for f in files if f.endswith('.mp3') or f.endswith('.wav')]
        audio_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
        
        self.audio_paths = [os.path.join(self.directory, f) for f in audio_files]
        
        with ThreadPoolExecutor(max_workers=-1) as executor:
            results = list(executor.map(self.process_single_audio, self.audio_paths))
        
        # Extracted features for all files
        common_features, age_features, gender_features, paths = zip(*results)
        print(paths)
        print(np.array(age_features).shape)
        print(np.array(gender_features).shape)
        print(np.array(common_features).shape)
        age_features = np.hstack((np.array(age_features),np.array(common_features)))
        gender_features = np.hstack((np.array(gender_features),np.array(common_features)))
        print(age_features.shape)
        print(gender_features.shape)

        return age_features, gender_features
        
    def process_single_audio(self, filepath):
        audio ,sr= librosa.load(filepath, sr=None,mono=True)
        audio = unsilenced_audio(audio)
        audio = normalize_loudness(audio)
        audio = nr.reduce_noise(y=audio, sr=sr)

        # Extract features
        common_features = extract_common_features(audio, sr=self.sr)
        age_feature = extract_age_features(audio, sr=self.sr)
        gender_feature = extract_gender_features(audio, sr=self.sr)

        # Return the features and the file path for joining
        return common_features, age_feature, gender_feature, filepath