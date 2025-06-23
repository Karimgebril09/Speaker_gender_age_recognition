import numpy as np
import librosa
from pydub import AudioSegment

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

def extract_common_features(audio,sr):
    pass    

def extract_age_features(audio, sr):
    pass

def extract_gender_features(audio, sr):
    pass