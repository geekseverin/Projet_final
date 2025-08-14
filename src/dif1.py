import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Dataset personnalisé pour charger les données prétraitées avec augmentation et normalisation
class BrainTumorDataset(Dataset):
    def __init__(self, processed_dir, timepoints=["Timepoint_1", "Timepoint_2", "Timepoint_5"], image_size=128):
        self.processed_dir = processed_dir
        self.timepoints = timepoints
        self.image_size = image_size
        self.samples = []

        for file in os.listdir(processed_dir):
            if file.endswith('_processed.pkl'):
                with open(os.path.join(processed_dir, file), 'rb') as f:
                    patient_data = pickle.load(f)
                patient_id = file.split('_')[1]
                for timepoint in timepoints:
                    if timepoint in patient_data:
                        image_data = patient_data[timepoint]['image']
                        mask_data = patient_data[timepoint]['mask']
                        depth = image_data.shape[2]
                        for z in range(depth // 4, 3 * depth // 4):
                            if np.sum(image_data[:, :, z]) > 0:  # slices non vides
                                self.samples.append({
                                    'image': image_data[:, :, z].astype(np.float32),
                                    'mask': mask_data[:, :, z].astype(np.float32),
                                    'patient_id': patient_id,
                                    'timepoint': timepoint,
                                    'slice_idx': z
                                })
        print(f"Dataset créé avec {len(self.samples)} échantillons")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample['image']
        mask = sample['mask']

        # Normalisation
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

        # Data augmentation simple
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=0)

        # Redimension
        if image.shape != (self.image_size, self.image_size):
            image = cv2.resize(image, (self.image_size, self.image_size))
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        return (
            torch.FloatTensor(image).unsqueeze(0),
            torch.FloatTensor(mask).unsqueeze(0),
            sample['patient_id'],
            sample['timepoint'],
            sample['slice_idx']
        )

# Modèle de diffusion amélioré
class DiffusionModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, hidden_channels=64):
        super(DiffusionModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels*2, hidden_channels*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels*4, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.encoder(x)

# Entraînement du modèle
class DiffusionSegmentationTrainer:
    def __init__(self, processed_dir, results_dir="results", batch_size=4, num_epochs=3, learning_rate=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = BrainTumorDataset(processed_dir)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.model = DiffusionModel().to(self.device)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.num_epochs = num_epochs

    def add_noise(self, image, mask, t):
        noise = torch.randn_like(image)
        t = t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        alpha = torch.sqrt(1 - t)  # meilleur schedule de diffusion
        noisy_image = alpha * image + (1 - alpha) * noise * mask
        return noisy_image, noise

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            for i, (images, masks, patient_ids, timepoints, slice_idxs) in enumerate(self.dataloader):
                images = images.to(self.device)
                masks = masks.to(self.device)
                self.optimizer.zero_grad()

                t = torch.rand(images.size(0), device=self.device)
                noisy_images, true_noise = self.add_noise(images, masks, t)

                pred_noise = self.model(noisy_images)
                loss = self.mse_loss(pred_noise, true_noise)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                if (i+1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.num_epochs}], Step [{i+1}/{len(self.dataloader)}], Loss: {loss.item():.4f}")

            avg_loss = running_loss / len(self.dataloader)
            print(f"Epoch [{epoch+1}/{self.num_epochs}] completed, Average Loss: {avg_loss:.4f}")

            # Sauvegarde des prédictions pour visualisation
            self.model.eval()
            with torch.no_grad():
                for i, (images, masks, patient_ids, timepoints, slice_idxs) in enumerate(self.dataloader):
                    if i == 0:
                        images = images.to(self.device)
                        t = torch.tensor([0.5], device=self.device).repeat(images.size(0))
                        noisy_images, _ = self.add_noise(images, masks.to(self.device), t)
                        pred_noise = self.model(noisy_images)
                        pred_mask = (noisy_images - pred_noise).clamp(0,1)
                        for j in range(min(3, images.size(0))):
                            plt.figure(figsize=(10,5))
                            plt.subplot(1,3,1)
                            plt.imshow(images[j,0].cpu().numpy(), cmap='gray')
                            plt.title(f"Image {patient_ids[j]}")
                            plt.axis('off')
                            plt.subplot(1,3,2)
                            plt.imshow(masks[j,0].cpu().numpy(), cmap='gray')
                            plt.title(f"Mask {timepoints[j]}")
                            plt.axis('off')
                            plt.subplot(1,3,3)
                            plt.imshow(pred_mask[j,0].cpu().numpy(), cmap='gray')
                            plt.title(f"Pred Mask Slice {slice_idxs[j]}")
                            plt.axis('off')
                            plt.savefig(os.path.join(self.results_dir, f"pred_{patient_ids[j]}_{timepoints[j]}_{slice_idxs[j]}.png"))
                            plt.close()

        torch.save(self.model.state_dict(), os.path.join(self.results_dir, "diffusion_model.pth"))
        print(f"Modèle sauvegardé dans {self.results_dir}/diffusion_model.pth")

if __name__ == "__main__":
    processed_directory = "/home/severin/Bureau/projet_final/processed_data"
    trainer = DiffusionSegmentationTrainer(processed_directory)
    trainer.train()
