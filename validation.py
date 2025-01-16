import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass
from torch.optim import Optimizer
from dataloader import create_fma_small_dataloader
from model import DenoisingAutoencoder, UNetGenerator, Discriminator
from collections import defaultdict
import torchaudio
import torchaudio.transforms as T

seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



class AudioProcessor:
    def __init__(self):
        self.spec_transform = T.Spectrogram(
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            power=2.0
        )
        self.griffinlim_transform = T.GriffinLim(
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            power=2.0
        )

    def _get_spectrogram(self, waveform):
        # Generate spectrogram
        spec = self.spec_transform(waveform)
        # Logarithm transformation
        spec = torch.log1p(spec)
        # Normalization
        spec = (spec - spec.mean()) / (spec.std() + 1e-8)
        return spec

    def _inverse_spectrogram(self, spec):
        # Inverse Normalization
        # spec = spec * (spec.std() + 1e-8) + spec.mean()
        # Inverse Logarithm transformation
        spec = torch.expm1(spec)
        # Inverse transformation using Griffin-Lim
        waveform = self.griffinlim_transform(spec)
        return waveform

# Initialize the audio processor
processor = AudioProcessor()




@dataclass
class TrainingConfig:
    """Configurations for training"""
    gen_lambda: float = 10.0  # Generator reconstruction loss weight
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelWrapper:
    """Model wrapper for managing models and optimizers"""

    def __init__(self,
                 denoiser,
                 generator,
                 discriminator,
                 denoiser_optimizer: Optimizer,
                 gen_optimizer: Optimizer,
                 disc_optimizer: Optimizer,
                 config: TrainingConfig):
        self.denoiser = denoiser
        self.generator = generator
        self.discriminator = discriminator
        self.denoiser_optimizer = denoiser_optimizer
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.config = config

    def train(self):
        """Set the models to train mode"""
        self.denoiser.train()
        self.generator.train()
        self.discriminator.train()

    def eval(self):
        """Set the models to evaluation mode"""
        self.denoiser.eval()
        self.generator.eval()
        self.discriminator.eval()

    def to(self, device):
        """Move the models to the given device"""
        self.denoiser.to(device)
        self.generator.to(device)
        self.discriminator.to(device)


class AudioDenoiser:
    """Audio denoiser trainer"""

    def __init__(self, model_wrapper: ModelWrapper):
        self.model = model_wrapper
        self.model.to(device=model_wrapper.config.device)
        self.device = model_wrapper.config.device


    @torch.no_grad()
    def evaluate_testset(self, test_loader) -> Dict[str, float]:
        """Evaluate the model

        Args:
            test_loader: test data loader

        Returns:
            Includes a dictionary with multiple evaluation metrics including MSE, denoiser MSE, and generator MSE
        """
        self.model.eval()
        metrics = {
            'total_mse': 0,
            'denoiser_mse': 0,
            'generator_mse': 0
        }
        num_batches = 0

        for batch in test_loader:
            noisy_audio = batch['noisy_spec'].to(self.device)
            clean_audio = batch['clean_spec'].to(self.device)

            # Denoiser MSE
            denoised = self.model.denoiser(noisy_audio)
            denoiser_mse = F.mse_loss(denoised, clean_audio)
            metrics['denoiser_mse'] += denoiser_mse.item()

            # Generator MSE
            restored = self.model.generator(denoised)
            generator_mse = F.mse_loss(restored, clean_audio)
            metrics['generator_mse'] += generator_mse.item()

            # Total MSE
            metrics['total_mse'] += generator_mse.item()
            num_batches += 1

        # Calculate average metrics
        for key in metrics:
            metrics[key] /= num_batches

        self.model.train()
        return metrics


def create_trainer(denoiser, generator, discriminator,
                   learning_rate: float = 0.0002,
                   gen_lambda: float = 10.0) -> AudioDenoiser:

    """Create the configuration for trainer"""
    config = TrainingConfig(gen_lambda=gen_lambda)

    # Create optimizers
    denoiser_optimizer = torch.optim.Adam(denoiser.parameters(), lr=learning_rate)
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

    # Create model wrapper
    model_wrapper = ModelWrapper(
        denoiser, generator, discriminator,
        denoiser_optimizer, gen_optimizer, disc_optimizer,
        config
    )

    return AudioDenoiser(model_wrapper)


# Example usage
if __name__ == '__main__':


    # Initialize the models
    denoiser = DenoisingAutoencoder()
    generator = UNetGenerator()

    denoiser_model_path = './denoiser_wave_100.pt'
    denoiser.load_state_dict(torch.load(denoiser_model_path))
    denoiser.eval()
    denoiser.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    generator_model_path = './generator_wave_100.pt'
    generator.load_state_dict(torch.load(generator_model_path))
    generator.eval()
    generator.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Create the loader
    train_loader, val_loader, test_loader = create_fma_small_dataloader(
        base_dir=r'./',
        batch_size=5,
        num_workers=4,
        duration=3
    )


    for batch_idx, batch in enumerate(test_loader):
        noisy_audio = batch['noisy_wave']
        clean_audio = batch['clean_wave']
        silenced_audio = batch['silenced_wave']
        track_id = batch['track_id']

        noisy_audio = noisy_audio.unsqueeze(1)
        clean_audio = clean_audio.unsqueeze(1)
        silenced_audio = silenced_audio.unsqueeze(1)

        noisy_audio = noisy_audio.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        clean_audio = clean_audio.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        silenced_audio = silenced_audio.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # Calculate the denoisor MSE
        denoised = denoiser(noisy_audio)
        restruction_audio = generator(denoised)
        denoiser_mse = F.mse_loss(denoised, clean_audio)
        
        for i in range(len(track_id)):
            torchaudio.save(f'./results/denoised_{track_id[i]}.wav', denoised[i].reshape(1,-1).detach().cpu(), 16000)
            torchaudio.save(f'./results/noisy_{track_id[i]}.wav', noisy_audio[i].reshape(1,-1).detach().cpu(), 16000)
            torchaudio.save(f'./results/clean_{track_id[i]}.wav', clean_audio[i].reshape(1,-1).detach().cpu(), 16000)
            torchaudio.save(f'./results/restruction_{track_id[i]}.wav', restruction_audio[i].reshape(1,-1).detach().cpu(), 16000)


        print(denoiser_mse.item())





