# src/autoencoder.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenoisingAutoEncoder3D(nn.Module):
    """
    AutoEncoder 3D débruitant.
    forward(x) -> (reconstructed, latent)
    - latent: vecteur (B, latent_dim)
    - reconstructed: tensor (B, C, D, H, W) redimensionné pour correspondre à x
    """
    def __init__(self, in_channels=1, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder (4 blocs stride=2)
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),

            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),

            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),

            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),
        )

        # Projection vers latent (on pool puis flatten)
        self.encoder_linear = nn.Sequential(
            nn.AdaptiveAvgPool3d((4, 4, 4)),  # fixe la grille latente
            nn.Flatten(),
            nn.Linear(256 * 4 * 4 * 4, latent_dim),
            nn.ReLU(inplace=True)
        )

        # Décoder depuis latent
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4 * 4),
            nn.ReLU(inplace=True)
        )

        # Bloc de déconv (symétrique à l'encoder)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),

            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),

            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),

            nn.ConvTranspose3d(32, in_channels, kernel_size=4, stride=2, padding=1),
            # PAS de Sigmoid — on laisse la sortie libre pour correspondre à la normalisation d'entrée (z-score etc.)
        )

    def encode(self, x):
        feats = self.encoder(x)
        latent = self.encoder_linear(feats)
        return latent

    def decode(self, latent):
        x = self.decoder_linear(latent)
        x = x.view(-1, 256, 4, 4, 4)
        x = self.decoder(x)
        return x

    def forward(self, x):
        """
        x: (B, C, D, H, W)
        returns: (reconstructed, latent)
        reconstructed is resized to x.shape[2:]
        """
        B = x.size(0)
        latent = self.encode(x)                       # (B, latent_dim)
        reconstructed = self.decode(latent)          # (B, C, d', h', w') usually (64,64,64)
        # Redimensionner pour correspondre exactement à l'entrée
        target_size = x.shape[2:]  # (D, H, W)
        reconstructed = F.interpolate(reconstructed, size=target_size, mode='trilinear', align_corners=False)
        return reconstructed, latent

    def add_noise(self, x, noise_factor=0.1):
        """Ajoute bruit gaussien sans clamp — conserve la dynamique (utile si tes images sont normalisées en z-score)."""
        noise = torch.randn_like(x) * noise_factor
        noisy_x = x + noise
        return noisy_x


# Trainer léger pour l'AE (utilisé dans train_autoencoder)
class AutoEncoderTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()

    def train_step(self, clean_images, optimizer, noise_factor=0.1):
        self.model.train()
        optimizer.zero_grad()
        noisy_images = self.model.add_noise(clean_images, noise_factor)
        reconstructed, latent = self.model(noisy_images)
        loss = self.criterion(reconstructed, clean_images)
        loss.backward()
        optimizer.step()
        return loss.item()

    def validate_step(self, clean_images, noise_factor=0.1):
        self.model.eval()
        with torch.no_grad():
            noisy_images = self.model.add_noise(clean_images, noise_factor)
            reconstructed, latent = self.model(noisy_images)
            loss = self.criterion(reconstructed, clean_images)
        return loss.item(), reconstructed

    def denoise_image(self, noisy_image):
        self.model.eval()
        with torch.no_grad():
            denoised, latent = self.model(noisy_image)
        return denoised, latent


# Fonction utilitaire d'entraînement (optionnelle) — garde la même interface que précédemment
def train_autoencoder(model, train_loader, val_loader, num_epochs=50, lr=1e-3, output_dir="outputs"):
    """
    Entraîne l'AE et sauvegarde une courbe loss (PNG + CSV) dans output_dir/logs.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import csv
    import os

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = AutoEncoderTrainer(model, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    train_losses, val_losses = [], []
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        sum_loss = 0.0; nb = 0
        for batch in train_loader:
            images = batch['input_images'][:, 0].to(device)  # premier timepoint
            loss = trainer.train_step(images, optimizer, noise_factor=0.15)
            sum_loss += loss; nb += 1
        avg_train = sum_loss / max(1, nb)

        # validation
        model.eval()
        sum_val = 0.0; nbv = 0
        for batch in val_loader:
            images = batch['input_images'][:, 0].to(device)
            l, _ = trainer.validate_step(images, noise_factor=0.15)
            sum_val += l; nbv += 1
        avg_val = sum_val / max(1, nbv)

        scheduler.step(avg_val)
        train_losses.append(avg_train); val_losses.append(avg_val)
        print(f"[AE] Epoch {epoch+1}/{num_epochs} | Train: {avg_train:.6f} | Val: {avg_val:.6f}")

        # sauvegarde courbe + csv
        plt.figure(figsize=(6,4))
        plt.plot(range(1, len(train_losses)+1), train_losses, marker='o', label='Train')
        plt.plot(range(1, len(val_losses)+1), val_losses, marker='o', label='Val')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "logs", "ae_loss_curve.png"), dpi=150)
        plt.close()
        with open(os.path.join(output_dir, "logs", "ae_loss_history.csv"), 'w', newline='') as f:
            writer = csv.writer(f); writer.writerow(["epoch", "train_loss", "val_loss"])
            for i, (tr, va) in enumerate(zip(train_losses, val_losses), start=1):
                writer.writerow([i, tr, va])

    return train_losses, val_losses


def create_autoencoder(latent_dim=128):
    return DenoisingAutoEncoder3D(in_channels=1, latent_dim=latent_dim)


# Petit test si on exécute directement
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_autoencoder(latent_dim=128).to(device)

    batch_size = 2
    # NOTE: ton ordering peut être (C, D, H, W) ou (C, D, H, W) selon tes données.
    # Ici on teste avec (B, C, D, H, W) = (2,1,128,128,64) comme dans ton message.
    test_input = torch.randn(batch_size, 1, 128, 128, 64).to(device)

    print("Input shape:", test_input.shape)
    reconstructed, latent = model(test_input)
    print("Reconstructed shape:", reconstructed.shape)
    print("Latent shape:", latent.shape)

    noisy_input = model.add_noise(test_input, noise_factor=0.2)
    denoised, _ = model(noisy_input)
    print("Denoised shape:", denoised.shape)
