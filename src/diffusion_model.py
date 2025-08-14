import os
import math
import random
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# =============================
# Config & chemins
# =============================
PROCESSED_DIR = "data/processed"
MASKS_DIR = "data/masks"
MODELS_DIR = "outputs/models"
LOGS_DIR = "outputs/logs"

os.makedirs(MASKS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32  # peut rester en FP32; l'AMP gère la précision mixte

# Taille volumes (doit matcher data_loader)
TARGET_SIZE = (64, 128, 128)

# Patchs 3D pour accélérer
PATCH_SIZE = (32, 64, 64)
# Lors de l'entraînement, on échantillonne des patchs aléatoires
PATCHES_PER_VOLUME = 4

# Diffusion
TIMESTEPS = 50  # réduit pour aller plus vite
BETA_START = 1e-4
BETA_END = 0.02

# Entraînement
BATCH_SIZE = 2  # grâce aux patchs on peut monter un peu
EPOCHS = 8
LR = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
MIXED_PRECISION = True

# Visualisation
VISUALIZE_EVERY = 1  # epochs

# =============================
# Utilitaires
# =============================

def center_crop_or_pad(vol, out_shape):
    """Centre crop/pad un volume numpy à out_shape."""
    z, y, x = vol.shape
    Z, Y, X = out_shape
    out = np.zeros(out_shape, dtype=vol.dtype)

    z0 = max(0, (Z - z) // 2)
    y0 = max(0, (Y - y) // 2)
    x0 = max(0, (X - x) // 2)

    zs = max(0, (z - Z) // 2)
    ys = max(0, (y - Y) // 2)
    xs = max(0, (x - X) // 2)

    out[z0:z0+min(Z, z), y0:y0+min(Y, y), x0:x0+min(X, x)] = vol[zs:zs+min(Z, z), ys:ys+min(Y, y), xs:xs+min(X, x)]
    return out


def random_patch_indices(shape, patch):
    """Retourne un indice (z,y,x) aléatoire valide pour un patch donné."""
    Z, Y, X = shape
    pZ, pY, pX = patch
    z = 0 if Z == pZ else random.randint(0, Z - pZ)
    y = 0 if Y == pY else random.randint(0, Y - pY)
    x = 0 if X == pX else random.randint(0, X - pX)
    return z, y, x


def extract_patch(vol, start, patch):
    z, y, x = start
    pZ, pY, pX = patch
    return vol[z:z+pZ, y:y+pY, x:x+pX]


def dice_coefficient(pred, target, eps=1e-6):
    """Dice sur volumes binaires torch (B,1,D,H,W)."""
    pred_b = (pred > 0.5).float()
    target_b = (target > 0.5).float()
    inter = (pred_b * target_b).sum(dim=(1,2,3,4))
    union = pred_b.sum(dim=(1,2,3,4)) + target_b.sum(dim=(1,2,3,4))
    dice = (2*inter + eps) / (union + eps)
    return dice.mean().item()


# =============================
# Dataset avec patchs 3D
# =============================
class TumorPatchDataset(Dataset):
    """
    Charge les volumes (cerveau + masque) puis renvoie PATCHES_PER_VOLUME patchs aléatoires.
    Chaque __getitem__ retourne UN patch (brain, mask) shape: (1,pZ,pY,pX)
    """
    def __init__(self, timepoints=("Timepoint_1", "Timepoint_2"), patches_per_volume=PATCHES_PER_VOLUME):
        self.samples = []
        self.patches_per_volume = patches_per_volume
        patients = [d for d in os.listdir(PROCESSED_DIR) if d.startswith("PatientID_")]
        for patient in patients:
            for tp in timepoints:
                base = os.path.join(PROCESSED_DIR, patient, tp)
                brain_path = os.path.join(base, f"{patient}_{tp}_brain_t1c.nii.gz")
                mask_path = os.path.join(base, f"{patient}_{tp}_tumorMask.nii.gz")
                if os.path.exists(brain_path) and os.path.exists(mask_path):
                    self.samples.append((brain_path, mask_path))

        # Précharge léger: on ne garde pas tout en RAM; on ne stocke que les chemins

    def __len__(self):
        return len(self.samples) * self.patches_per_volume

    def __getitem__(self, idx):
        vol_idx = idx // self.patches_per_volume
        brain_path, mask_path = self.samples[vol_idx]

        brain = nib.load(brain_path).get_fdata().astype(np.float32)
        mask = nib.load(mask_path).get_fdata().astype(np.float32)

        # Assure la taille cible (au cas où)
        if brain.shape != TARGET_SIZE:
            brain = center_crop_or_pad(brain, TARGET_SIZE)
        if mask.shape != TARGET_SIZE:
            mask = center_crop_or_pad(mask, TARGET_SIZE)

        # Focalisation sur régions tumorales si possible
        if mask.max() > 0:
            # échantillonner autour d'un voxel tumoral
            zc, yc, xc = np.argwhere(mask > 0)[np.random.randint((mask > 0).sum())]
            pZ, pY, pX = PATCH_SIZE
            z = np.clip(zc - pZ//2, 0, TARGET_SIZE[0] - pZ)
            y = np.clip(yc - pY//2, 0, TARGET_SIZE[1] - pY)
            x = np.clip(xc - pX//2, 0, TARGET_SIZE[2] - pX)
        else:
            z, y, x = random_patch_indices(TARGET_SIZE, PATCH_SIZE)

        brain_patch = extract_patch(brain, (z,y,x), PATCH_SIZE)
        mask_patch = extract_patch(mask, (z,y,x), PATCH_SIZE)

        brain_t = torch.from_numpy(brain_patch)[None, ...].to(DTYPE)
        mask_t = torch.from_numpy(mask_patch)[None, ...].to(DTYPE)
        return brain_t, mask_t


# =============================
# UNet3D léger + Embedding temporel
# =============================
class ConvBlock3D(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1),
            nn.GroupNorm(4, out_c),
            nn.SiLU(),
            nn.Conv3d(out_c, out_c, 3, padding=1),
            nn.GroupNorm(4, out_c),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=2, base=24, out_channels=1):
        super().__init__()
        self.enc1 = ConvBlock3D(in_channels, base)
        self.enc2 = ConvBlock3D(base, base*2)
        self.enc3 = ConvBlock3D(base*2, base*4)
        self.pool = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock3D(base*4, base*8)

        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.dec3 = ConvBlock3D(base*8 + base*4, base*4)
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.dec2 = ConvBlock3D(base*4 + base*2, base*2)
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.dec1 = ConvBlock3D(base*2 + base, base)

        self.out_conv = nn.Conv3d(base, out_channels, 1)

        # injection du temps (simple FiLM additif sur 4 niveaux)
        self.time_mlp = nn.Sequential(
            nn.Linear(64, base*8),
            nn.SiLU(),
            nn.Linear(base*8, base + base*2 + base*4 + base*8)
        )
        self.base = base

    def forward(self, x, t_embed):
        # t_embed: (B,64)
        tb = self.time_mlp(t_embed)  # (B, base + 2base + 4base + 8base)
        b1, b2, b3, b4 = torch.split(tb, [self.base, self.base*2, self.base*4, self.base*8], dim=1)

        # Encoder
        e1 = self.enc1(x)
        e1 = e1 + b1[:, :, None, None, None]
        e2 = self.enc2(self.pool(e1))
        e2 = e2 + b2[:, :, None, None, None]
        e3 = self.enc3(self.pool(e2))
        e3 = e3 + b3[:, :, None, None, None]

        # Bottleneck
        bn = self.bottleneck(self.pool(e3))
        bn = bn + b4[:, :, None, None, None]

        # Decoder
        d3 = self.dec3(torch.cat([self.up3(bn), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = self.out_conv(d1)
        return out


class TimeEmbedding(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        self.fc = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def forward(self, t):
        # sinusoidal embedding
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=t.device).float() / half
        )
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.fc(emb)


class DiffusionModel(nn.Module):
    def __init__(self, timesteps=TIMESTEPS):
        super().__init__()
        self.timesteps = timesteps
        self.unet = UNet3D(in_channels=2, base=24, out_channels=1)
        self.time_emb = TimeEmbedding(64)

        betas = torch.linspace(BETA_START, BETA_END, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1,1,1,1,1)
        sqrt_1m = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1,1)
        return sqrt_ac * x0 + sqrt_1m * noise, noise

    def forward(self, cond, x_t, t):
        t_emb = self.time_emb(t)
        x = torch.cat([cond, x_t], dim=1)
        return self.unet(x, t_emb)

    @torch.no_grad()
    def ddim_sample(self, cond, shape, steps=25, eta=0.0):
        """DDIM sampling pour aller plus vite (deterministic si eta=0)."""
        b = shape[0]
        device = cond.device
        ddim_steps = steps
        step_indices = torch.linspace(0, self.timesteps-1, ddim_steps, device=device).long()

        x = torch.randn(shape, device=device)
        for i in range(ddim_steps-1, -1, -1):
            t = step_indices[i]
            t_b = torch.full((b,), t, device=device, dtype=torch.long)
            eps = self.forward(cond, x, t_b)
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alphas_cumprod[t]
            if i > 0:
                t_prev = step_indices[i-1]
                alpha_bar_prev = self.alphas_cumprod[t_prev]
            else:
                alpha_bar_prev = torch.tensor(1.0, device=device)

            # equations DDIM
            x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
            dir_xt = torch.sqrt(1 - alpha_bar_prev) * eps
            x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt
            if eta > 0 and i > 0:
                sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t/alpha_bar_prev))
                x += sigma * torch.randn_like(x)
        return x


# =============================
# Entraînement & Validation
# =============================

def train():
    dataset = TumorPatchDataset(timepoints=("Timepoint_1", "Timepoint_2"))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    model = DiffusionModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    scaler = torch.cuda.amp.GradScaler(enabled=MIXED_PRECISION)

    best_loss = float('inf')
    for epoch in range(1, EPOCHS+1):
        model.train()
        running = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for brain, mask in pbar:
            brain = brain.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            t = torch.randint(0, TIMESTEPS, (brain.size(0),), device=DEVICE, dtype=torch.long)

            with torch.cuda.amp.autocast(enabled=MIXED_PRECISION):
                noisy_mask, noise = model.q_sample(mask, t)
                pred_noise = model(brain, noisy_mask, t)
                loss = loss_fn(pred_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if GRAD_CLIP_NORM is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss = running / max(1, len(loader))
        print(f"Epoch {epoch} mean loss: {epoch_loss:.4f}")

        # checkpoint
        ckpt_path = os.path.join(MODELS_DIR, f"diffusion_model_epoch{epoch}.pth")
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'loss': epoch_loss,
        }, ckpt_path)

        # Visualisation & validation simple (sur un seul volume complet si dispo)
        if epoch % VISUALIZE_EVERY == 0:
            try:
                visualize_and_log(model)
            except Exception as e:
                print("Visualisation erreur:", e)

        # garder le meilleur
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "diffusion_model_best.pth"))


@torch.no_grad()
def visualize_and_log(model):
    model.eval()

    # Prendre un patient Timepoint_1
    patients = [d for d in os.listdir(PROCESSED_DIR) if d.startswith("PatientID_")]
    if not patients:
        return
    patient = patients[0]
    tp = "Timepoint_1"

    brain_path = os.path.join(PROCESSED_DIR, patient, tp, f"{patient}_{tp}_brain_t1c.nii.gz")
    mask_path = os.path.join(PROCESSED_DIR, patient, tp, f"{patient}_{tp}_tumorMask.nii.gz")
    if not (os.path.exists(brain_path) and os.path.exists(mask_path)):
        return

    brain = nib.load(brain_path).get_fdata().astype(np.float32)
    mask = nib.load(mask_path).get_fdata().astype(np.float32)

    if brain.shape != TARGET_SIZE:
        brain = center_crop_or_pad(brain, TARGET_SIZE)
    if mask.shape != TARGET_SIZE:
        mask = center_crop_or_pad(mask, TARGET_SIZE)

    brain_t = torch.from_numpy(brain)[None, None, ...].to(DEVICE)

    # DDIM rapide
    pred_mask = model.ddim_sample(brain_t, brain_t.shape, steps=25, eta=0.0)
    pred_mask = pred_mask.clamp(-3, 3)  # bornage léger
    pred_mask_sig = torch.sigmoid(pred_mask)

    # Dice (avec seuillage 0.5)
    target_t = torch.from_numpy(mask)[None, None, ...].to(DEVICE)
    dice = dice_coefficient(pred_mask_sig, target_t)
    print(f"Validation Dice (seuillage 0.5) : {dice:.4f}")

    # Sauvegarde NIfTI + PNG
    pred_np = pred_mask_sig.squeeze().cpu().numpy()
    out_nii = os.path.join(MASKS_DIR, f"{patient}_{tp}_preview_pred.nii.gz")
    nib.save(nib.Nifti1Image(pred_np.astype(np.float32), np.eye(4)), out_nii)

    # figure 2D (slice centrale)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    zc = TARGET_SIZE[0] // 2
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(brain[zc], cmap='gray')
    axs[0].set_title('Image originale')
    axs[0].axis('off')
    axs[1].imshow(mask[zc], cmap='gray')
    axs[1].set_title('Masque réel')
    axs[1].axis('off')
    axs[2].imshow(pred_np[zc], cmap='gray')
    axs[2].set_title(f'Prédiction (Dice={dice:.2f})')
    axs[2].axis('off')
    plt.tight_layout()
    png_path = os.path.join(LOGS_DIR, f"preview_{patient}_{tp}.png")
    plt.savefig(png_path, dpi=150)
    plt.close(fig)


# =============================
# Inférence pour générer tous les masques
# =============================
@torch.no_grad()
def generate_masks(ckpt="diffusion_model_best.pth", steps=25):
    model = DiffusionModel().to(DEVICE)
    state = torch.load(os.path.join(MODELS_DIR, ckpt), map_location=DEVICE)
    if isinstance(state, dict) and 'model' in state:
        model.load_state_dict(state['model'])
    else:
        model.load_state_dict(state)
    model.eval()

    patients = [d for d in os.listdir(PROCESSED_DIR) if d.startswith("PatientID_")]
    for patient in patients:
        patient_mask_dir = os.path.join(MASKS_DIR, patient)
        os.makedirs(patient_mask_dir, exist_ok=True)
        for tp in ("Timepoint_1", "Timepoint_2", "Timepoint_5"):
            brain_path = os.path.join(PROCESSED_DIR, patient, tp, f"{patient}_{tp}_brain_t1c.nii.gz")
            if not os.path.exists(brain_path):
                continue
            brain = nib.load(brain_path).get_fdata().astype(np.float32)
            if brain.shape != TARGET_SIZE:
                brain = center_crop_or_pad(brain, TARGET_SIZE)
            brain_t = torch.from_numpy(brain)[None, None, ...].to(DEVICE)
            pred = model.ddim_sample(brain_t, brain_t.shape, steps=steps, eta=0.0)
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()

            # Binarisation optionnelle
            pred_bin = (pred > 0.5).astype(np.float32)

            nib.save(nib.Nifti1Image(pred.astype(np.float32), np.eye(4)),
                     os.path.join(patient_mask_dir, f"{patient}_{tp}_generated_mask_prob.nii.gz"))
            nib.save(nib.Nifti1Image(pred_bin, np.eye(4)),
                     os.path.join(patient_mask_dir, f"{patient}_{tp}_generated_mask_bin.nii.gz"))
            print(f"✅ Masques générés pour {patient} - {tp}")


if __name__ == "__main__":
    # Entraînement + aperçu
    train()
    # Inférence complète
    generate_masks()