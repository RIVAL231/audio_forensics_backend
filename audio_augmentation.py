# audio_augmentation.py

from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import numpy as np
import librosa
from sklearn.preprocessing import MinMaxScaler

# Define audio augmentation pipeline
audio_augmentations = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
])

def augment_audio(y, sr):
    """
    Applies the defined augmentations to the audio signal.
    """
    augmented_y = audio_augmentations(samples=y, sample_rate=sr)
    return augmented_y

def process_audio(file_path, sr=11000, n_mels=256, max_time_frames=256, apply_augmentation=True):
    """
    Process audio file with optional augmentation and return mel spectrogram.
    """
    try:
        y, original_sr = librosa.load(file_path, sr=sr)

        if apply_augmentation:
            y = augment_audio(y, sr)

        # Compute Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Truncate or pad to max_time_frames
        if mel_spectrogram.shape[1] > max_time_frames:
            mel_spectrogram = mel_spectrogram[:, :max_time_frames]
        else:
            pad_width = max_time_frames - mel_spectrogram.shape[1]
            mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant')

        # Normalize the spectrogram
        scaler = MinMaxScaler()
        mel_spectrogram_normalized = scaler.fit_transform(mel_spectrogram)
        
        # Add channel dimension
        mel_spectrogram_normalized = mel_spectrogram_normalized[..., np.newaxis]

        return mel_spectrogram_normalized
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None