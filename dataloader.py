import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import librosa
import pickle

seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False







class FMASmallDataset(Dataset):
    def __init__(self,
                 base_dir,
                 split='training',
                 sample_rate=16000,
                 duration=3):
        self.base_dir = os.path.abspath(base_dir)
        self.audio_dir = os.path.join(self.base_dir, 'fma_small')
        self.metadata_dir = os.path.join(self.base_dir, 'fma_metadata')
        self.sample_rate = sample_rate
        self.duration = duration
        self.segment_length = int(sample_rate * duration)

        self._check_directories()
        self.tracks = self._load_metadata()
        self.tracks = self._split_dataset(split)

        self.spec_transform = T.Spectrogram(
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            power=2.0
        )

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_mels=80
        )

        print(f"Loaded {len(self.tracks)} tracks for {split}")

    def _check_directories(self):
        if not os.path.exists(self.audio_dir):
            raise ValueError(f"Audio directory not found: {self.audio_dir}")
        if not os.path.exists(self.metadata_dir):
            raise ValueError(f"Metadata directory not found: {self.metadata_dir}")

        tracks_file = os.path.join(self.metadata_dir, 'tracks.csv')
        if not os.path.exists(tracks_file):
            raise ValueError(f"tracks.csv not found in {self.metadata_dir}")

    def _load_metadata(self):
        try:
            tracks_file = os.path.join(self.metadata_dir, 'tracks.csv')
            tracks = pd.read_csv(tracks_file, index_col=0, header=[0, 1])

            small = tracks['set', 'subset'] <= 'small'
            tracks = tracks[small][[('track', 'title'), ('track', 'genre_top')]]
            tracks.columns = ['title', 'genre']

            valid_tracks = []
            for track_id in tracks.index:
                audio_path = self._get_audio_path(track_id)
                try:
                    # waveform, sr = torchaudio.load(audio_path)
                    waveform, sr = librosa.load(audio_path, sr=None)
                    waveform = torch.tensor(waveform)
                    if not torch.all(waveform == 0) and not torch.isnan(waveform).any():
                        valid_tracks.append(track_id)
                except:
                    continue

            tracks = tracks.loc[valid_tracks]
            tracks = tracks.dropna()

            if len(tracks) == 0:
                raise ValueError("No valid tracks found after filtering")

            return tracks

        except Exception as e:
            print(f"Error loading metadata: {str(e)}")
            if 'tracks' in locals():
                print("Available columns:", tracks.columns.tolist())
            raise

    def _get_audio_path(self, track_id):
        track_id = int(track_id)
        tid_str = '{:06d}'.format(track_id)
        folder = tid_str[:3]
        filename = f"{tid_str}.mp3"
        return os.path.join(self.audio_dir, folder, filename)

    def _load_audio(self, track_id):
        """Load audio file and return a segment of the specified duration"""
        filepath = self._get_audio_path(track_id)

        # waveform, sr = torchaudio.load(filepath)
        waveform, sr = librosa.load(filepath, sr=None)
        waveform = torch.tensor(waveform)
        waveform = waveform.reshape(1,-1)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        if torch.all(waveform == 0) or torch.isnan(waveform).any():
            raise RuntimeError("Corrupted or silent audio file")

        if waveform.shape[1] > self.segment_length:
            start = torch.randint(0, waveform.shape[1] - self.segment_length, (1,))
            waveform = waveform[:, start:start + self.segment_length]
        else:
            num_repeats = (self.segment_length + waveform.shape[1] - 1) // waveform.shape[1]
            waveform = waveform.repeat(1, num_repeats)
            waveform = waveform[:, :self.segment_length]

        return waveform

    def _split_dataset(self, split):
        n_tracks = len(self.tracks)
        indices = np.random.permutation(n_tracks)

        if split == 'training':
            split_indices = indices[:int(0.8 * n_tracks)]
        elif split == 'validation':
            split_indices = indices[int(0.8 * n_tracks):int(0.9 * n_tracks)]
        else:  # test
            split_indices = indices[int(0.9 * n_tracks):]

        return self.tracks.iloc[split_indices]

    def _add_noise(self, waveform, snr_db=20):
        signal_power = torch.mean(waveform ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        return waveform + noise


    def _add_silience(self, waveform, num_silence_segments = 1, max_silence_duration = 1, min_silence_duration = 0.5):
        
        for _ in range(num_silence_segments):
            # Random silence duration
            silence_duration = np.random.uniform(min_silence_duration, max_silence_duration)
            silence_samples = int(silence_duration * self.sample_rate)
            
            # Random silence position
            start_position = np.random.randint(0, waveform.shape[1] - silence_samples)
            end_position = start_position + silence_samples
            
            # Silencing the waveform
            silenced_waveform = waveform.clone()
            silenced_waveform[:,start_position:end_position] = 0
        
        return silenced_waveform




    def _get_spectrogram(self, waveform):
        spec = self.spec_transform(waveform)
        spec = torch.log1p(spec)
        spec = (spec - spec.mean()) / (spec.std() + 1e-8)
        return spec

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        """Get a batch of data, jump to the next file if an error occurs"""
        while idx < len(self):
            try:
                track_id = self.tracks.index[idx]
                genre = self.tracks.iloc[idx]['genre']

                clean_wave = self._load_audio(track_id)
                silenced_wave = self._add_silience(clean_wave)
                noisy_wave = self._add_noise(silenced_wave)

                clean_spec = self._get_spectrogram(clean_wave)
                noisy_spec = self._get_spectrogram(noisy_wave)

                return {
                    'track_id': track_id,
                    'genre': genre,
                    'silenced_wave': silenced_wave,
                    'noisy_wave': noisy_wave,
                    'clean_wave': clean_wave,
                    'noisy_spec': noisy_spec,
                    'clean_spec': clean_spec
                }
            except Exception as e:
                print(f"Error loading track {track_id}: {str(e)}")
                idx += 1  # Move to the next track

        # If no valid data was found, then start over
        return self.__getitem__(0)


def create_fma_small_dataloader(base_dir, batch_size=32, num_workers=4, **kwargs):
    # train_dataset = FMASmallDataset(base_dir, split='training', **kwargs)
    # val_dataset = FMASmallDataset(base_dir, split='validation', **kwargs)
    # test_dataset = FMASmallDataset(base_dir, split='test', **kwargs)

    # with open('/hhd/datasets/msy/train_set.pkl', "wb") as f:
    #     pickle.dump(train_dataset, f)
    # with open('/hhd/datasets/msy/val_set.pkl', "wb") as f:
    #     pickle.dump(val_dataset, f)
    # with open('/hhd/datasets/msy/test_set.pkl', "wb") as f:
    #     pickle.dump(test_dataset, f)

    with open('/hhd/datasets/msy/train_set.pkl', "rb") as f:
        train_dataset = pickle.load(f)
    with open('/hhd/datasets/msy/val_set.pkl', "rb") as f:
        val_dataset = pickle.load(f)
    with open('/hhd/datasets/msy/test_set.pkl', "rb") as f:
        test_dataset = pickle.load(f)


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


# if __name__ == '__main__':
#     base_dir = 'D:\\cloud\\fma_metadata'
#
#     try:
#         train_loader, val_loader, test_loader = create_fma_small_dataloader(
#             base_dir=base_dir,
#             batch_size=32,
#             num_workers=4,
#             duration=3
#         )
#
#         # Test data loading
#         for batch in train_loader:
#             print("Batch size:", len(batch['track_id']))
#             print("Audio shape:", batch['clean_wave'].shape)
#             print("Spectrogram shape:", batch['clean_spec'].shape)
#             print("Genres:", batch['genre'])
#             break
#
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         import traceback
#
#         traceback.print_exc()