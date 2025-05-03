import os
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings("ignore")
from utils_audio import extract_features_preprocessed,apply_tranformations

class DataProcessor:
    def __init__(self, directory, sr=None):
        self.directory = directory
        self.sr = sr
        self.audio_paths = []
        
    def load_all_data(self):
        files = os.listdir(self.directory)
        audio_files = [f for f in files if f.endswith('.mp3') or f.endswith('.wav')]
        audio_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
        print("found audio files:", len(audio_files))
        self.audio_paths = [os.path.join(self.directory, f) for f in audio_files]
        results=None
        start=time.time()
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(executor.map(self.process_single_audio, self.audio_paths))
        end=time.time()
        print("Time taken to process all audio files:", end-start)        

        df=pd.DataFrame(results)

        start=time.time()
        apply_tranformations(df)
        end=time.time()
        print("Time taken to apply transformations:", end-start)
        # print(df.columns)
        #result is a list of dictionaries, each containing the features for one audio file
        return df

    def process_single_audio(self, filepath):
        # Load the audio file
        features = extract_features_preprocessed(filepath)
        return features