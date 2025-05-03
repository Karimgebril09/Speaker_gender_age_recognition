import os
import librosa
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings("ignore")
from utils_audio import  unsilenced_audio, normalize_loudness, extract_features_preprocessed,apply_tranformations

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
        results=None
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(self.process_single_audio, self.audio_paths))
        
        df=pd.DataFrame(results)

        apply_tranformations(df)

        # print(df.columns)
        #result is a list of dictionaries, each containing the features for one audio file
        return df

    def process_single_audio(self, filepath):
        # Load the audio file
        features = extract_features_preprocessed(filepath)
        return features