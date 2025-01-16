import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass
from torch.optim import Optimizer
from dataloader import create_fma_small_dataloader
from model import DenoisingAutoencoder, UNetGenerator, Discriminator
from collections import defaultdict


seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


@dataclass
class TrainingConfig:
    """Training configuration dataclass"""
    gen_lambda: float = 1.0  # generator loss weight
    device: torch.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


class ModelWrapper:
    """Model wrapper, managing the models and optimizers"""

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
        """Set the models to training mode"""
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

    def _train_denoiser(self, noisy_audio: torch.Tensor,
                        clean_audio: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Train the denoiser model"""
        self.model.denoiser_optimizer.zero_grad()


        denoised = self.model.denoiser(noisy_audio)
        loss = F.mse_loss(denoised, clean_audio)
        loss.backward()
        self.model.denoiser_optimizer.step()
        return denoised, loss.item()

    def _train_discriminator(self, clean_audio: torch.Tensor,
                             gen_output: torch.Tensor) -> float:
        """Train the discriminator"""
        self.model.disc_optimizer.zero_grad()

        disc_real = self.model.discriminator(clean_audio)
        disc_fake = self.model.discriminator(gen_output.detach())

        # Calculate the loss between real and generated samples
        # real_loss = F.binary_cross_entropy(
        #     disc_real,
        #     torch.ones_like(disc_real)
        # )
        # fake_loss = F.binary_cross_entropy(
        #     disc_fake,
        #     torch.zeros_like(disc_fake)
        # )
        real_loss = torch.mean(F.relu(1.0 - disc_real))
        fake_loss = torch.mean(F.relu(1.0 + disc_fake))

        disc_loss = (real_loss + fake_loss) / 2
        disc_loss.backward()
        self.model.disc_optimizer.step()

        return disc_loss.item()

    def _train_generator(self, denoised: torch.Tensor,
                         clean_audio: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Train the generator model"""
        self.model.gen_optimizer.zero_grad()

        gen_output = self.model.generator(denoised.detach())
        disc_output = self.model.discriminator(gen_output)

        # Calculate the adversarial and reconstruction losses
        adv_loss = F.binary_cross_entropy(
            disc_output,
            torch.ones_like(disc_output)
        )
        recon_loss = F.l1_loss(gen_output, clean_audio)

        # Total Loss
        gen_loss = adv_loss + self.model.config.gen_lambda * recon_loss
        gen_loss.backward()
        self.model.gen_optimizer.step()

        return gen_output, {
            'gen_loss': gen_loss.item(),
            'recon_loss': recon_loss.item(),
            'adv_loss': adv_loss.item()
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a training step"""
        #noisy_audio = batch['noisy_spec'].to(self.device)
        #clean_audio = batch['clean_spec'].to(self.device)
        noisy_audio = batch['noisy_wave'].to(self.device)
        clean_audio = batch['clean_wave'].to(self.device)
        silenced_audio = batch['silenced_wave'].to(self.device)

        noisy_audio = noisy_audio.unsqueeze(1)
        clean_audio = clean_audio.unsqueeze(1)
        silenced_audio = silenced_audio.unsqueeze(1)

        # Train the denoiser
        denoised, denoiser_loss = self._train_denoiser(noisy_audio, silenced_audio)

        # Train the GAN
        gen_output = self.model.generator(denoised.detach())
        disc_loss = self._train_discriminator(clean_audio, gen_output)
        _, gen_losses = self._train_generator(denoised, clean_audio)

        return {
            'denoiser_loss': denoiser_loss,
            'disc_loss': disc_loss,
            **gen_losses
        }

    @torch.no_grad()
    def evaluate(self, val_loader) -> float:
        """evaluate the model on the validation set"""
        self.model.eval()
        total_mse = 0
        num_batches = 0

        for batch in val_loader:
            #noisy_audio = batch['noisy_spec'].to(self.device)
            #clean_audio = batch['clean_spec'].to(self.device)
            noisy_audio = batch['noisy_wave'].to(self.device)
            clean_audio = batch['clean_wave'].to(self.device)
            silenced_audio = batch['silenced_wave'].to(self.device)

            noisy_audio = noisy_audio.unsqueeze(1)
            clean_audio = clean_audio.unsqueeze(1)
            silenced_audio = silenced_audio.unsqueeze(1)

            denoised = self.model.denoiser(noisy_audio)
            restored = self.model.generator(denoised)

            mse = F.mse_loss(restored, clean_audio)
            total_mse += mse.item()
            num_batches += 1

        self.model.train()
        return total_mse / num_batches

    @torch.no_grad()
    def evaluate_testset(self, test_loader) -> Dict[str, float]:
        """Evaluate the model on the test set

        Args:
            test_loader: test data loader

        Returns:
            Includes multiple evaluation metrics in a dictionary, including MSE, denoiser MSE, and generator MSE
        """
        self.model.eval()
        metrics = {
            'total_mse': 0,
            'denoiser_mse': 0,
            'generator_mse': 0
        }
        num_batches = 0

        for batch in test_loader:
            #noisy_audio = batch['noisy_spec'].to(self.device)
            #clean_audio = batch['clean_spec'].to(self.device)
            noisy_audio = batch['noisy_wave'].to(self.device)
            clean_audio = batch['clean_wave'].to(self.device)
            silenced_audio = batch['silenced_wave'].to(self.device)

            noisy_audio = noisy_audio.unsqueeze(1)
            clean_audio = clean_audio.unsqueeze(1)
            silenced_audio = silenced_audio.unsqueeze(1)

            # Calculate denoiser MSE
            denoised = self.model.denoiser(noisy_audio)
            denoiser_mse = F.mse_loss(denoised, silenced_audio)
            metrics['denoiser_mse'] += denoiser_mse.item()

            # Calculate generator MSE
            restored = self.model.generator(denoised)
            generator_mse = F.mse_loss(restored, clean_audio)
            metrics['generator_mse'] += generator_mse.item()

            # Calculate total MSE
            metrics['total_mse'] += generator_mse.item()
            num_batches += 1

        # Calculate average metrics
        for key in metrics:
            metrics[key] /= num_batches

        self.model.train()
        return metrics


def create_trainer(denoiser, generator, discriminator,
                   learning_rate: float = 0.0002,
                   gen_lambda: float = 1.0) -> AudioDenoiser:

    """Create the configuration for trainer"""
    config = TrainingConfig(gen_lambda=gen_lambda)

    # Create optimizers
    denoiser_optimizer = torch.optim.Adam(denoiser.parameters(), lr=learning_rate)
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate*0.8)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate*0.1)

    # Create the model wrapper
    model_wrapper = ModelWrapper(
        denoiser, generator, discriminator,
        denoiser_optimizer, gen_optimizer, disc_optimizer,
        config
    )

    return AudioDenoiser(model_wrapper)


# Example
if __name__ == '__main__':


    # Initialize the models
    denoiser = DenoisingAutoencoder()
    generator = UNetGenerator()
    discriminator = Discriminator()

    # Create the trainer
    trainer = create_trainer(denoiser, generator, discriminator)

    # Create the data loaders
    train_loader, val_loader, test_loader = create_fma_small_dataloader(
        base_dir=r'/hhd/datasets/msy',
        batch_size=32,
        num_workers=4,
        duration=3
    )

    # best_val_loss = 1000
    # Train Loop
    num_epochs = 100
    for epoch in range(num_epochs):
        epoch_losses = defaultdict(float)
        num_batches = 0

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        for batch_idx, batch in enumerate(train_loader):
            losses = trainer.train_step(batch)

            # Accumulate the losses
            for key, value in losses.items():
                epoch_losses[key] += value
            num_batches += 1

            if (batch_idx + 1) % 100 == 0:
                print(f"Process: {batch_idx + 1}/{len(train_loader)} batches")

        # Calculate the average losses
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}

        # Evaluation
        val_mse = trainer.evaluate(val_loader)

        # Print the losses
        print("\nCurrent State:")
        print(f"{'Loss Type':<20} {'Loss Value':>12}")
        print("-" * 35)
        for loss_name, loss_value in avg_losses.items():
            print(f"{loss_name:<20} {loss_value:>12.6f}")
        print(f"{'validation_mse':<20} {val_mse:>12.6f}")
        print("=" * 50)

        # if best_val_loss == 0 or val_mse < best_val_loss:
        #     best_val_loss = val_mse
        #     print("Save Model...")
        #     torch.save(trainer.model.denoiser.state_dict(), "denoiser.pt")
        #     torch.save(trainer.model.generator.state_dict(), "generator.pt")
        #     torch.save(trainer.model.discriminator.state_dict(), "discriminator.pt")

        torch.save(trainer.model.denoiser.state_dict(), "/hhd/datasets/msy/denoiser_wave_final.pt")
        torch.save(trainer.model.generator.state_dict(), "/hhd/datasets/msy/generator_wave_final.pt")
        torch.save(trainer.model.discriminator.state_dict(), "/hhd/datasets/msy/discriminator_wave_final.pt")

    # Evaluation on test set
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate_testset(test_loader)
    print("\nEvaluation Result:")
    print("-" * 35)
    for metric_name, metric_value in test_metrics.items():
        print(f"{metric_name:<20} {metric_value:>12.6f}")
    print("=" * 50)