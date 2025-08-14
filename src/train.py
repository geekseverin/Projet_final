# src/train.py
import os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# Imports depuis nos modules
from autoencoder import create_autoencoder, train_autoencoder
from convlstm import create_tumor_predictor, TumorEvolutionPredictor, ConvLSTM3D
# diffusion (optionnel) - si tu as un module diffusion_model.py adapté
try:
    from diffusion_model import DiffusionModel, DiffusionSegmentationTrainer, create_diffusion_model
    HAS_DIFFUSION = True
except Exception:
    HAS_DIFFUSION = False

# create_data_loaders : tu dois avoir cette fonction dans data_loader.py
try:
    from data_loader import create_data_loaders
except Exception:
    create_data_loaders = None


def plot_training_curves(train_losses, val_losses, title, save_path):
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(train_losses)+1), train_losses, marker='o', label='Train')
    plt.plot(range(1, len(val_losses)+1), val_losses, marker='o', label='Val')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train_complete_pipeline(data_path, output_dir, config):
    """
    Entraîne : AutoEncoder (optionnel) puis ConvLSTM predictor.
    Exige create_data_loaders(data_path, batch_size, train_split) -> train_loader, val_loader
    Chaque batch doit contenir:
      - batch['input_images']: (B, T, C, D, H, W)
      - batch['input_masks']:  (B, T, 1, D, H, W)
      - batch['target_mask']:  (B, 1, D, H, W)  # target pour T3
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

    if create_data_loaders is None:
        raise RuntimeError("La fonction create_data_loaders n'a pas été trouvée.\n"
                           "Ajoute dans data_loader.py une fonction create_data_loaders(data_path, batch_size, train_split) "
                           "retournant (train_loader, val_loader) avec les clés 'input_images','input_masks','target_mask'.")

    # Chargement données
    train_loader, val_loader = create_data_loaders(data_path,
                                                  batch_size=config['batch_size'],
                                                  train_split=config['train_split'])
    print("Dataloaders chargés.")

    # 1) AutoEncoder (optionnel)
    autoencoder = create_autoencoder(latent_dim=config['latent_dim']).to(device)
    if config.get('train_autoencoder', True):
        print("Entraînement AutoEncoder...")
        ae_train_losses, ae_val_losses = train_autoencoder(autoencoder, train_loader, val_loader,
                                                           num_epochs=config['ae_epochs'], lr=config['ae_lr'],
                                                           output_dir=output_dir)
        torch.save(autoencoder.state_dict(), os.path.join(output_dir, "models", "autoencoder.pth"))
        plot_training_curves(ae_train_losses, ae_val_losses, "AutoEncoder Loss",
                             os.path.join(output_dir, "logs", "ae_loss_curve.png"))
    else:
        # essayer de charger un AE pré-entraîné s'il existe
        ae_path = os.path.join(output_dir, "models", "autoencoder.pth")
        if os.path.exists(ae_path):
            autoencoder.load_state_dict(torch.load(ae_path, map_location=device))
            print("AutoEncoder chargé depuis:", ae_path)
        else:
            print("Attention: autoencoder non entraîné et aucun checkpoint trouvé.")

    # 2) ConvLSTM predictor
    print("\nEntraînement du modèle de prédiction (ConvLSTM)...")
    predictor = create_tumor_predictor(autoencoder,
                                      latent_dim=config['latent_dim'],
                                      use_mask=config.get('use_mask', True),
                                      mask_channels=1,
                                      hidden_dims=config.get('pred_hidden_dims', [64, 32]),
                                      kernel_sizes=config.get('pred_kernel_sizes', [(3,3,3),(3,3,3)]),
                                      num_layers=len(config.get('pred_hidden_dims', [64,32])),
                                      grid_size=(4,4,4))
    predictor = predictor.to(device)

    # Trainer minimal
    trainer = None
    # on crée notre propre boucle d'entraînement (BCE + Dice)
    criterion_bce = nn.BCELoss()
    def dice_loss(pred, target, smooth=1e-6):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        inter = (pred_flat * target_flat).sum()
        return 1 - ((2. * inter + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))

    optimizer = torch.optim.Adam(predictor.parameters(), lr=config['pred_lr'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience = config.get('patience', 10)
    patience_counter = 0

    for epoch in range(config['pred_epochs']):
        predictor.train()
        sum_loss = 0.0; nb = 0
        pbar = tqdm(train_loader, desc=f"Pred Epoch {epoch+1}/{config['pred_epochs']}")
        for batch in pbar:
            inputs = batch['input_images'].to(device)   # (B,T,C,D,H,W)
            masks_in = batch['input_masks'].to(device)  # (B,T,1,D,H,W)
            target = batch['target_mask'].to(device)    # (B,1,D,H,W)

            optimizer.zero_grad()
            pred = predictor(inputs, masks_in)

            bce = criterion_bce(pred, target)
            dice = dice_loss(pred, target)
            total = config['loss_alpha'] * bce + (1 - config['loss_alpha']) * dice

            total.backward()
            optimizer.step()

            sum_loss += total.item(); nb += 1
            pbar.set_postfix({'loss': f"{total.item():.4f}", 'bce': f"{bce.item():.4f}", 'dice': f"{dice:.4f}"})

        avg_train = sum_loss / max(1, nb)
        train_losses.append(avg_train)

        # validation
        predictor.eval()
        sum_val = 0.0; nbv = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input_images'].to(device)
                masks_in = batch['input_masks'].to(device)
                target = batch['target_mask'].to(device)
                pred = predictor(inputs, masks_in)
                bce = criterion_bce(pred, target)
                dice = dice_loss(pred, target)
                total = config['loss_alpha'] * bce + (1 - config['loss_alpha']) * dice
                sum_val += total.item(); nbv += 1
        avg_val = sum_val / max(1, nbv)
        val_losses.append(avg_val)

        print(f"[Epoch {epoch+1}] Train: {avg_train:.6f} | Val: {avg_val:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # scheduler
        scheduler.step(avg_val)

        # sauvegarde du meilleur
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': predictor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val
            }, os.path.join(output_dir, "models", "best_predictor.pth"))
            print("Meilleur modèle sauvegardé.")
        else:
            patience_counter += 1

        # courbes sauvegarde
        plot_training_curves(train_losses, val_losses, "Prediction Training Loss",
                             os.path.join(output_dir, "logs", "prediction_loss_curve.png"))
        # csv
        import csv
        with open(os.path.join(output_dir, "logs", "prediction_loss_history.csv"), 'w', newline='') as f:
            writer = csv.writer(f); writer.writerow(['epoch','train_loss','val_loss'])
            for i, (tr, va) in enumerate(zip(train_losses, val_losses), start=1):
                writer.writerow([i, tr, va])

        if patience_counter >= patience:
            print(f"Early stopping (patience={patience})")
            break

    # sauvegarde final
    torch.save({
        'model_state_dict': predictor.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses
    }, os.path.join(output_dir, "models", "final_predictor.pth"))
    print("Entraînement prédiction terminé. Meilleure val loss:", best_val_loss)
    return predictor, train_losses, val_losses


def get_default_config():
    return {
        'batch_size': 2,
        'train_split': 0.8,
        'patience': 10,
        'train_autoencoder': True,
        'latent_dim': 128,
        'ae_epochs': 20,
        'ae_lr': 1e-3,
        'pred_epochs': 50,
        'pred_lr': 1e-4,
        'loss_alpha': 0.7,
        'pred_hidden_dims': [64, 32],
        'pred_kernel_sizes': [(3,3,3),(3,3,3)]
    }


if __name__ == "__main__":
    data_path = "data/processed"  # ou data/raw selon ton create_data_loaders
    output_dir = "outputs"
    config = get_default_config()
    os.makedirs(output_dir, exist_ok=True)

    predictor, train_losses, val_losses = train_complete_pipeline(data_path, output_dir, config)
